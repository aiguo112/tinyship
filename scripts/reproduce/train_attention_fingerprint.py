#!/usr/bin/env python
"""
Attention-pooled Sub-Center ArcFace retrain (the ONE gated retrain).

Architecture (leakage-aware):
  encoder (same MobileNetV3-Small 0.1M) -> clip emb
  gated attention over a MIXED-SESSION clip-set -> set emb
  Sub-Center ArcFace (K=3) on the set emb

Set construction is part of the method:
  TRAIN: sample w clips of one vessel preferring DISTINCT sub_init/date so
         attention learns identity, not within-recording continuity.
  EVAL:  disjoint enroll-set vs probe-sets; both attention-pooled.
         Primary cell mirrors free_experiments best: w≈3, k_enroll=20.
         Pre-registered gate: EER <= 0.08 vs free-exp floor 0.114.

Saves encoder + attention + clip embeddings BEFORE scoring.

  python scripts/reproduce/train_attention_fingerprint.py --dry-run
  python scripts/reproduce/train_attention_fingerprint.py
  python scripts/reproduce/score_attention_fingerprint.py
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.frontend import extract, in_channels  # noqa: E402
from src.models.fingerprint import (  # noqa: E402
    GatedAttentionPool, SubCenterArcFaceHead, build_encoder,
)

SPLIT_DIR = ROOT / "data" / "processed" / "splits"
CACHE = ROOT / "data" / "processed" / "feat_cache"
REPORTS = ROOT / "reports"
COLS = {"path": "file_path", "mmsi": "MMSI", "session": "sub_init", "date": "date"}
FEATURE = "cqt_mfcc"
EMB_DIM, WIDTH = 128, 0.25
FREE_EXP_BASELINE = 0.1143
GATE_EER = 0.08


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_split(name):
    p = SPLIT_DIR / f"mmsi_{name}.csv"
    if not p.exists():
        sys.exit(f"missing split file {p}")
    df = pd.read_csv(p).rename(columns={v: k for k, v in COLS.items()})
    df = df[df["mmsi"].astype(str) != "0"].copy()
    df["mmsi"] = df["mmsi"].astype(str)
    df["session"] = df["session"].astype(str)
    df["date"] = df["date"].astype(str)
    return df.reset_index(drop=True)


def load_background(name):
    p = SPLIT_DIR / f"mmsi_{name}.csv"
    df = pd.read_csv(p).rename(columns={v: k for k, v in COLS.items()})
    return df[df["mmsi"].astype(str) == "0"].copy().reset_index(drop=True)


def load_feat(path: str):
    key = CACHE / f"{FEATURE}_{hashlib.sha256(path.encode()).hexdigest()[:16]}.npy"
    if key.exists():
        return np.load(key)
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    x = extract(y, sr or 32000, FEATURE)
    np.save(key, x)
    return x


def _lr_at(epoch: int, epochs: int, base_lr: float, warmup: int = 5) -> float:
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    t = (epoch - warmup) / max(epochs - warmup, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * t))


def build_ship_index(df: pd.DataFrame):
    """Per-ship lists of row indices, grouped by session for mixed sampling."""
    ships = {}
    for mmsi, g in df.groupby("mmsi"):
        by_sess = {}
        for s, sg in g.groupby("session"):
            by_sess[str(s)] = sg.index.to_numpy()
        ships[str(mmsi)] = {
            "sessions": by_sess,
            "all": g.index.to_numpy(),
            "dates": g["date"].to_numpy(),
        }
    return ships


def sample_mixed_set(ship_rec, w: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw w clip indices preferring DISTINCT sessions (leakage-aware).
    Falls back to any clips if fewer than w sessions exist.
    """
    sessions = list(ship_rec["sessions"].keys())
    rng.shuffle(sessions)
    picked = []
    # round-robin across sessions so we don't empty one recording first
    sess_pools = {s: list(ship_rec["sessions"][s]) for s in sessions}
    for s in sess_pools:
        rng.shuffle(sess_pools[s])
    while len(picked) < w:
        progressed = False
        for s in sessions:
            if sess_pools[s]:
                picked.append(sess_pools[s].pop())
                progressed = True
                if len(picked) >= w:
                    break
        if not progressed:
            # exhaust: sample with replacement from all
            all_idx = ship_rec["all"]
            need = w - len(picked)
            extra = rng.choice(all_idx, size=need, replace=(len(all_idx) < need))
            picked.extend(list(np.atleast_1d(extra)))
            break
    return np.asarray(picked[:w], dtype=np.int64)


def eer(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    if len(neg) > 50 * len(pos):
        neg = np.random.default_rng(0).choice(neg, size=50 * len(pos), replace=False)
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    thr = np.quantile(s, np.linspace(0, 1, 1500)) if len(np.unique(s)) > 1500 else np.unique(s)
    best = (1.0, 0.5)
    for t in thr:
        p = s >= t
        far = float(np.mean(p[y == 0])) if (y == 0).any() else 0.0
        frr = float(np.mean(~p[y == 1])) if (y == 1).any() else 0.0
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return float(best[1])


def attention_pool_numpy(attn_mod, emb_stack, device):
    """emb_stack: (W, D) or (B, W, D) numpy -> pooled (B, D), attn (B, W)."""
    import torch
    attn_mod.eval()
    with torch.no_grad():
        H = torch.from_numpy(emb_stack.astype(np.float32)).to(device)
        if H.dim() == 2:
            H = H.unsqueeze(0)
        pooled, a = attn_mod(H)
        return pooled.cpu().numpy(), a.cpu().numpy()


def evaluate_attention(E, ids, dates, attn_mod, device, k_enroll=20, probe_w=3, seed=0):
    """
    Disjoint enroll-set vs probe-sets, both attention-pooled.
    Comparable to free_experiments cell (k=20, w=3 mean) but with learned attn.
    Also reports mean-prototype ablation on the same split.
    """
    ids = np.asarray(ids).astype(str)
    dates = np.asarray(dates).astype(str) if dates is not None else None
    rng = np.random.default_rng(seed)
    ships = [s for s in np.unique(ids) if (ids == s).sum() >= k_enroll + probe_w]
    if len(ships) < 2:
        return {"note": "too few ships", "usable_ships": len(ships)}

    enroll_proto_attn, enroll_proto_mean, ship_list = {}, {}, []
    probe_sets = []  # (pooled_attn, pooled_mean, ship_id)

    for s in ships:
        idx = np.where(ids == s)[0].copy()
        rng.shuffle(idx)
        gal = idx[:k_enroll]
        rest = idx[k_enroll:]
        # enroll
        pa, _ = attention_pool_numpy(attn_mod, E[gal], device)
        enroll_proto_attn[s] = pa[0]
        m = E[gal].mean(0); m = m / (np.linalg.norm(m) + 1e-9)
        enroll_proto_mean[s] = m
        ship_list.append(s)
        # probe sets of size probe_w (disjoint from enroll)
        for start in range(0, len(rest) - probe_w + 1, probe_w):
            chunk = rest[start:start + probe_w]
            qa, _ = attention_pool_numpy(attn_mod, E[chunk], device)
            qm = E[chunk].mean(0); qm = qm / (np.linalg.norm(qm) + 1e-9)
            probe_sets.append((qa[0], qm, s))

    Pid = np.array(ship_list)
    P_attn = np.stack([enroll_proto_attn[s] for s in ship_list])
    P_mean = np.stack([enroll_proto_mean[s] for s in ship_list])

    def _scores(P, use_attn_probe=True):
        pos, neg = [], []
        for qa, qm, sid in probe_sets:
            v = qa if use_attn_probe else qm
            sims = P @ v
            pos.append(float(sims[Pid == sid][0]))
            neg.extend(sims[Pid != sid].tolist())
        return eer(np.array(pos), np.array(neg))

    # across_date secondary (tiny N): enroll on dates A, probe sets on dates B
    ad_pos, ad_neg, ad_ships = [], [], 0
    if dates is not None:
        for s in ships:
            idx = np.where(ids == s)[0]
            uniq = np.unique(dates[idx])
            if len(uniq) < 2:
                continue
            rng.shuffle(uniq)
            g_dates = set(uniq[: max(1, len(uniq) // 2)])
            g = idx[np.isin(dates[idx], list(g_dates))]
            p = idx[~np.isin(dates[idx], list(g_dates))]
            if len(g) < max(3, k_enroll // 4) or len(p) < probe_w:
                continue
            rng.shuffle(g); rng.shuffle(p)
            g = g[: min(len(g), k_enroll)]
            pa, _ = attention_pool_numpy(attn_mod, E[g], device)
            proto = pa[0]
            ad_ships += 1
            for start in range(0, len(p) - probe_w + 1, probe_w):
                chunk = p[start:start + probe_w]
                qa, _ = attention_pool_numpy(attn_mod, E[chunk], device)
                # score vs all attn enroll prototypes (rebuild with this ship's date-split proto)
                # Use ship_list prototypes but replace this ship's with date-split one
                Ptmp = P_attn.copy()
                Ptmp[Pid == s] = proto
                sims = Ptmp @ qa[0]
                ad_pos.append(float(sims[Pid == s][0]))
                ad_neg.extend(sims[Pid != s].tolist())

    out = {
        "protocol": "attention_set_disjoint",
        "k_enroll": k_enroll,
        "probe_w": probe_w,
        "usable_ships": len(ships),
        "n_probe_sets": len(probe_sets),
        "eer_attention": round(_scores(P_attn, True), 4),
        "eer_mean_ablation": round(_scores(P_mean, False), 4),
        "free_exp_baseline": FREE_EXP_BASELINE,
        "gate_eer": GATE_EER,
        "across_date": {
            "usable_ships": ad_ships,
            "eer_attention": round(eer(np.array(ad_pos), np.array(ad_neg)), 4) if ad_pos else None,
            "note": "secondary; tiny N on VTUAD test",
        },
    }
    out["gate_pass"] = bool(out["eer_attention"] <= GATE_EER)
    out["beats_free_exp"] = bool(out["eer_attention"] < FREE_EXP_BASELINE)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32, help="number of vessel-sets per batch")
    ap.add_argument("--set-w", type=int, default=5, help="clips per training set")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--scale", type=float, default=16.0)
    ap.add_argument("--K", type=int, default=3, help="sub-centers per identity")
    ap.add_argument("--k-enroll", type=int, default=20)
    ap.add_argument("--probe-w", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--init-ckpt", type=str, default="",
                    help="optional encoder warm-start (default: fingerprint_arcface_ckpt.pt if present)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tr, te = load_split("train"), load_split("test")
    bg_te = load_background("test")
    ship_idx = build_ship_index(tr)
    classes = sorted(ship_idx.keys())
    cmap = {m: i for i, m in enumerate(classes)}

    # How often can we actually mix sessions?
    mixable = sum(1 for s in classes if len(ship_idx[s]["sessions"]) >= a.set_w)
    date_multi = te.groupby("mmsi")["date"].nunique()
    log(f"train ships {len(classes)} clips {len(tr)} | set_w={a.set_w} "
        f"ships_with>={a.set_w}_sessions={mixable}/{len(classes)}")
    log(f"test ships {te['mmsi'].nunique()} clips {len(te)} | "
        f">=2 dates: {(date_multi >= 2).sum()} (across_date secondary)")
    log(f"eval cell: k_enroll={a.k_enroll} probe_w={a.probe_w} | "
        f"gate EER<={GATE_EER} vs free_exp {FREE_EXP_BASELINE}")

    if a.dry_run:
        # sampler smoke: one mixed set per ship
        rng = np.random.default_rng(0)
        n_sess = []
        for m in classes[:20]:
            idxs = sample_mixed_set(ship_idx[m], a.set_w, rng)
            sess = tr.loc[idxs, "session"].nunique()
            n_sess.append(sess)
        log(f"sampler smoke (first 20 ships): mean unique sessions/set={np.mean(n_sess):.2f}")
        log("DRY RUN only."); return

    import torch
    import torch.nn.functional as F

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        log("WARNING: no CUDA")
    CACHE.mkdir(parents=True, exist_ok=True)

    enc = build_encoder(in_channels(FEATURE), EMB_DIM, WIDTH).to(dev)
    attn = GatedAttentionPool(EMB_DIM, hidden=64).to(dev)
    head = SubCenterArcFaceHead(EMB_DIM, len(classes), K=a.K, s=a.scale, m=a.margin).to(dev)

    init_path = Path(a.init_ckpt) if a.init_ckpt else REPORTS / "fingerprint_arcface_ckpt.pt"
    if init_path.exists():
        ck = torch.load(init_path, map_location=dev, weights_only=False)
        missing, unexpected = enc.load_state_dict(ck["encoder"], strict=False)
        log(f"warm-start encoder from {init_path} (missing={len(missing)} unexpected={len(unexpected)})")
    else:
        log("no init ckpt; training encoder from scratch")

    opt = torch.optim.Adam(
        list(enc.parameters()) + list(attn.parameters()) + list(head.parameters()),
        lr=a.lr,
    )

    paths = tr["path"].astype(str).to_numpy()
    log(f"train attention+subcenter: epochs={a.epochs} K={a.K} set_w={a.set_w} "
        f"bs={a.bs} steps/ep={a.steps_per_epoch} lr={a.lr}")

    rng = np.random.default_rng(0)
    for ep in range(1, a.epochs + 1):
        lr = _lr_at(ep, a.epochs, a.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        enc.train(); attn.train(); head.train()
        running = 0.0
        for step in range(a.steps_per_epoch):
            # sample bs vessels (with replacement, class-balanced by uniform ship draw)
            batch_ships = rng.choice(classes, size=a.bs, replace=True)
            feats, labels = [], []
            for m in batch_ships:
                idxs = sample_mixed_set(ship_idx[m], a.set_w, rng)
                clips = [load_feat(paths[i]) for i in idxs]
                feats.append(np.stack(clips))
                labels.append(cmap[m])
            x = torch.from_numpy(np.stack(feats).astype(np.float32)).to(dev)  # (B, W, C, H, T)
            y = torch.tensor(labels, dtype=torch.long, device=dev)
            B, W = x.shape[0], x.shape[1]
            # encode each clip independently
            flat = x.view(B * W, *x.shape[2:])
            emb = enc(flat).view(B, W, -1)
            pooled, _ = attn(emb)
            loss = F.cross_entropy(head(pooled, y), y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item())
        if ep % 5 == 0 or ep == 1 or ep == a.epochs:
            log(f"epoch {ep}/{a.epochs} loss {running / a.steps_per_epoch:.4f} lr={lr:.2e}")

    # ---- embed all test clips (encoder only); save BEFORE score ----
    @torch.no_grad()
    def embed_df(df):
        enc.eval()
        out = []
        bs = 256
        plist = df["path"].astype(str).tolist()
        for i in range(0, len(plist), bs):
            chunk = [load_feat(p) for p in plist[i:i + bs]]
            x = torch.from_numpy(np.stack(chunk).astype(np.float32)).to(dev)
            out.append(enc(x).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, EMB_DIM), np.float32)

    te = te.reset_index(drop=True)
    E = embed_df(te)
    ids = te["mmsi"].astype(str).to_numpy()
    sess = te["session"].astype(str).to_numpy()
    dates = te["date"].astype(str).to_numpy()
    BG = embed_df(bg_te) if len(bg_te) else np.zeros((0, EMB_DIM), np.float32)

    REPORTS.mkdir(exist_ok=True)
    ckpt = REPORTS / "fingerprint_attention_ckpt.pt"
    torch.save(
        {
            "encoder": enc.state_dict(),
            "attention": attn.state_dict(),
            "head": head.state_dict(),
            "classes": classes,
            "config": {
                "epochs": a.epochs, "K": a.K, "set_w": a.set_w,
                "margin": a.margin, "scale": a.scale, "lr": a.lr,
                "k_enroll": a.k_enroll, "probe_w": a.probe_w,
                "emb_dim": EMB_DIM, "width": WIDTH, "feature": FEATURE,
            },
        },
        ckpt,
    )
    embp = REPORTS / "fingerprint_attention_embeddings.npz"
    np.savez(embp, E=E, ids=ids, sess=sess, dates=dates, BG=BG)
    log(f"saved weights -> {ckpt}")
    log(f"saved embeddings -> {embp}")

    try:
        rep = evaluate_attention(
            E, ids, dates, attn, dev,
            k_enroll=a.k_enroll, probe_w=a.probe_w, seed=0,
        )
    except Exception as e:
        log(f"scoring failed ({e}); ckpt+embeddings SAFE. "
            "python scripts/reproduce/score_attention_fingerprint.py")
        raise

    rep["_config"] = {
        "epochs": a.epochs, "K": a.K, "set_w": a.set_w,
        "margin": a.margin, "scale": a.scale, "lr": a.lr,
        "k_enroll": a.k_enroll, "probe_w": a.probe_w,
    }
    outp = REPORTS / "fingerprint_attention.json"
    outp.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    log(f"wrote {outp}")
    log(f"RESULT [attention] eer_attention={rep.get('eer_attention')} "
        f"eer_mean_ablation={rep.get('eer_mean_ablation')} "
        f"ships={rep.get('usable_ships')} n_probe_sets={rep.get('n_probe_sets')} "
        f"| across_date={rep.get('across_date')} "
        f"| beats_free_exp={rep.get('beats_free_exp')} gate_pass(<={GATE_EER})={rep.get('gate_pass')}")


if __name__ == "__main__":
    main()
