#!/usr/bin/env python
"""
E1b — train learned temporal aggregators over 1 s clip embeddings.

Modes:
  frozen   — train aggregator + SubCenterArcFaceHead on TRAIN embeddings;
             encoder NOT in optimizer (default).
  finetune — warm-start encoder from attention ckpt; train encoder + aggregator
             + head on spectrogram features.

Saves ckpt (and finetune embeddings) BEFORE scoring.

  python scripts/reproduce/train_e1b_aggregators.py --agg all --mode frozen
  python scripts/reproduce/train_e1b_aggregators.py --agg all --mode finetune
  python scripts/reproduce/train_e1b_aggregators.py --score-only
"""
from __future__ import annotations
import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.frontend import extract, in_channels  # noqa: E402
from src.models.fingerprint import (  # noqa: E402
    SubCenterArcFaceHead, build_encoder, build_temporal_aggregator,
)

SPLIT_DIR = ROOT / "data" / "processed" / "splits"
CACHE = ROOT / "data" / "processed" / "feat_cache"
REPORTS = ROOT / "reports"
COLS = {"path": "file_path", "mmsi": "MMSI", "session": "sub_init", "date": "date"}
FEATURE = "cqt_mfcc"
EMB_DIM, WIDTH = 128, 0.25
ENC_CKPT = REPORTS / "fingerprint_attention_ckpt.pt"
TRAIN_EMB = REPORTS / "fingerprint_attention_train_embeddings.npz"
TEST_EMB = REPORTS / "fingerprint_attention_embeddings.npz"
RUNNER = ROOT / "scripts" / "reproduce" / "run_e1b_temporal.py"
AGGS = ("gru", "tcn", "mha")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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
            all_idx = ship_rec["all"]
            need = w - len(picked)
            extra = rng.choice(all_idx, size=need, replace=(len(all_idx) < need))
            picked.extend(list(np.atleast_1d(extra)))
            break
    return np.asarray(picked[:w], dtype=np.int64)


def _lr_at(epoch: int, epochs: int, base_lr: float, warmup: int = 5) -> float:
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    t = (epoch - warmup) / max(epochs - warmup, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * t))


def embed_df(df, enc, device, bs=256):
    import torch

    enc.eval()
    out = []
    plist = df["path"].astype(str).tolist()
    log(f"embedding {len(plist)} clips (bs={bs})")
    with torch.no_grad():
        for i in range(0, len(plist), bs):
            chunk = [load_feat(p) for p in plist[i:i + bs]]
            x = torch.from_numpy(np.stack(chunk).astype(np.float32)).to(device)
            out.append(enc(x).cpu().numpy())
            if i == 0 or (i // bs) % 40 == 0:
                log(f"  embed {min(i + bs, len(plist))}/{len(plist)}")
    return np.concatenate(out) if out else np.zeros((0, EMB_DIM), np.float32)


def save_train_embeddings(tr, E):
    REPORTS.mkdir(exist_ok=True)
    np.savez(
        TRAIN_EMB,
        E=E,
        ids=tr["mmsi"].astype(str).to_numpy(),
        sess=tr["session"].astype(str).to_numpy(),
        dates=tr["date"].astype(str).to_numpy(),
    )
    log(f"saved train embeddings -> {TRAIN_EMB}  E={E.shape}")


def load_or_embed_train(device):
    import torch

    tr = load_split("train")
    if TRAIN_EMB.exists():
        d = np.load(TRAIN_EMB, allow_pickle=True)
        E = d["E"]
        ids = np.asarray(d["ids"]).astype(str)
        if len(E) != len(tr):
            sys.exit(
                f"BLOCKER: {TRAIN_EMB} has {len(E)} rows, train split has {len(tr)}. "
                "Delete the npz and re-embed."
            )
        if not np.array_equal(ids, tr["mmsi"].astype(str).to_numpy()):
            sys.exit(
                f"BLOCKER: train embedding ids do not match mmsi_train.csv. "
                f"Delete {TRAIN_EMB} and re-embed."
            )
        log(f"loaded existing train embeddings {TRAIN_EMB} E={E.shape}")
        return tr, E

    if not ENC_CKPT.exists():
        sys.exit(f"BLOCKER: missing frozen encoder {ENC_CKPT}")

    enc = build_encoder(in_channels(FEATURE), EMB_DIM, WIDTH).to(device)
    ck = torch.load(ENC_CKPT, map_location=device, weights_only=False)
    enc.load_state_dict(ck["encoder"], strict=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    log(f"loaded FROZEN encoder from {ENC_CKPT}")

    E = embed_df(tr, enc, device)
    save_train_embeddings(tr, E)
    del enc
    if device == "cuda":
        torch.cuda.empty_cache()
    return tr, E


def resolve_aggs(name: str):
    if name == "all":
        return list(AGGS)
    if name not in AGGS:
        sys.exit(f"unknown --agg {name!r}; expected gru|tcn|mha|all")
    return [name]


def ckpt_path(agg: str, mode: str) -> Path:
    return REPORTS / f"e1b_{agg}_{mode}_ckpt.pt"


def finetune_emb_path(agg: str) -> Path:
    return REPORTS / f"e1b_{agg}_finetune_embeddings.npz"


def log_mixed_sampler_stats(tr, ship_idx, classes, set_w: int):
    mixable = sum(1 for s in classes if len(ship_idx[s]["sessions"]) >= set_w)
    log(
        f"train ships {len(classes)} clips {len(tr)} | set_w={set_w} "
        f"ships_with>={set_w}_sessions={mixable}/{len(classes)}"
    )
    rng_smoke = np.random.default_rng(0)
    n_sess = []
    for m in classes:
        idxs = sample_mixed_set(ship_idx[m], set_w, rng_smoke)
        n_sess.append(tr.loc[idxs, "session"].nunique())
    log(
        f"mixed-session sampler: mean unique sessions/set={np.mean(n_sess):.2f} "
        f"(all {len(classes)} ships, set_w={set_w})"
    )


def train_frozen(tr, E, agg_name: str, args, device):
    import torch
    import torch.nn.functional as F

    ship_idx = build_ship_index(tr)
    classes = sorted(ship_idx.keys())
    cmap = {m: i for i, m in enumerate(classes)}
    log_mixed_sampler_stats(tr, ship_idx, classes, args.set_w)

    agg = build_temporal_aggregator(agg_name, EMB_DIM).to(device)
    head = SubCenterArcFaceHead(
        EMB_DIM, len(classes), K=args.K, s=args.scale, m=args.margin,
    ).to(device)
    opt = torch.optim.Adam(
        list(agg.parameters()) + list(head.parameters()), lr=args.lr,
    )
    opt_n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    agg_n = sum(p.numel() for p in agg.parameters())
    head_n = sum(p.numel() for p in head.parameters())
    if opt_n != agg_n + head_n:
        sys.exit(f"BLOCKER: optimizer n={opt_n} != agg+head {agg_n + head_n}")

    log(
        f"train {agg_name} FROZEN (encoder not in optimizer): epochs={args.epochs} "
        f"K={args.K} set_w={args.set_w} bs={args.bs} steps/ep={args.steps_per_epoch} "
        f"lr={args.lr}"
    )

    rng = np.random.default_rng(0)
    E32 = E.astype(np.float32)
    for ep in range(1, args.epochs + 1):
        agg.train()
        head.train()
        running = 0.0
        for _step in range(args.steps_per_epoch):
            batch_ships = rng.choice(classes, size=args.bs, replace=True)
            Hs, labels = [], []
            for m in batch_ships:
                idxs = sample_mixed_set(ship_idx[m], args.set_w, rng)
                Hs.append(E32[idxs])
                labels.append(cmap[m])
            H = torch.from_numpy(np.stack(Hs)).to(device)
            H = F.normalize(H, dim=-1)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            pooled = agg(H)
            if isinstance(pooled, tuple):
                pooled = pooled[0]
            loss = F.cross_entropy(head(pooled, y), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            log(f"[{agg_name}/frozen] epoch {ep}/{args.epochs} "
                f"loss {running / args.steps_per_epoch:.4f}")

    cfg = {
        "agg": agg_name,
        "mode": "frozen",
        "epochs": args.epochs,
        "K": args.K,
        "set_w": args.set_w,
        "margin": args.margin,
        "scale": args.scale,
        "lr": args.lr,
        "bs": args.bs,
        "steps_per_epoch": args.steps_per_epoch,
        "emb_dim": EMB_DIM,
        "encoder_frozen": True,
        "encoder_ckpt": str(ENC_CKPT.relative_to(ROOT)).replace("\\", "/"),
        "feature": FEATURE,
        "width": WIDTH,
    }
    out = ckpt_path(agg_name, "frozen")
    REPORTS.mkdir(exist_ok=True)
    torch.save(
        {
            "aggregator": agg.state_dict(),
            "head": head.state_dict(),
            "classes": classes,
            "config": cfg,
        },
        out,
    )
    log(f"saved {agg_name} frozen weights BEFORE scoring -> {out}")
    return agg


def train_finetune(agg_name: str, args, device):
    import torch
    import torch.nn.functional as F

    tr = load_split("train")
    te = load_split("test")
    bg_te = load_background("test")
    ship_idx = build_ship_index(tr)
    classes = sorted(ship_idx.keys())
    cmap = {m: i for i, m in enumerate(classes)}
    log_mixed_sampler_stats(tr, ship_idx, classes, args.set_w)

    if not ENC_CKPT.exists():
        sys.exit(f"BLOCKER: missing warm-start encoder {ENC_CKPT}")

    enc = build_encoder(in_channels(FEATURE), EMB_DIM, WIDTH).to(device)
    agg = build_temporal_aggregator(agg_name, EMB_DIM).to(device)
    head = SubCenterArcFaceHead(
        EMB_DIM, len(classes), K=args.K, s=args.scale, m=args.margin,
    ).to(device)

    ck = torch.load(ENC_CKPT, map_location=device, weights_only=False)
    missing, unexpected = enc.load_state_dict(ck["encoder"], strict=False)
    log(
        f"warm-start encoder from {ENC_CKPT} "
        f"(missing={len(missing)} unexpected={len(unexpected)})"
    )

    opt = torch.optim.Adam(
        list(enc.parameters()) + list(agg.parameters()) + list(head.parameters()),
        lr=args.lr,
    )
    paths = tr["path"].astype(str).to_numpy()
    log(
        f"train {agg_name} FINETUNE (encoder+agg+head): epochs={args.epochs} "
        f"K={args.K} set_w={args.set_w} bs={args.bs} steps/ep={args.steps_per_epoch} "
        f"lr={args.lr}"
    )

    rng = np.random.default_rng(0)
    for ep in range(1, args.epochs + 1):
        lr = _lr_at(ep, args.epochs, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        enc.train()
        agg.train()
        head.train()
        running = 0.0
        for _step in range(args.steps_per_epoch):
            batch_ships = rng.choice(classes, size=args.bs, replace=True)
            feats, labels = [], []
            for m in batch_ships:
                idxs = sample_mixed_set(ship_idx[m], args.set_w, rng)
                clips = [load_feat(paths[i]) for i in idxs]
                feats.append(np.stack(clips))
                labels.append(cmap[m])
            x = torch.from_numpy(np.stack(feats).astype(np.float32)).to(device)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            B, W = x.shape[0], x.shape[1]
            flat = x.view(B * W, *x.shape[2:])
            emb = enc(flat).view(B, W, -1)
            pooled = agg(emb)
            if isinstance(pooled, tuple):
                pooled = pooled[0]
            loss = F.cross_entropy(head(pooled, y), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            log(
                f"[{agg_name}/finetune] epoch {ep}/{args.epochs} "
                f"loss {running / args.steps_per_epoch:.4f} lr={lr:.2e}"
            )

    # embed test BEFORE scoring
    te = te.reset_index(drop=True)
    E = embed_df(te, enc, device)
    ids = te["mmsi"].astype(str).to_numpy()
    sess = te["session"].astype(str).to_numpy()
    dates = te["date"].astype(str).to_numpy()
    BG = embed_df(bg_te, enc, device) if len(bg_te) else np.zeros((0, EMB_DIM), np.float32)

    cfg = {
        "agg": agg_name,
        "mode": "finetune",
        "epochs": args.epochs,
        "K": args.K,
        "set_w": args.set_w,
        "margin": args.margin,
        "scale": args.scale,
        "lr": args.lr,
        "bs": args.bs,
        "steps_per_epoch": args.steps_per_epoch,
        "emb_dim": EMB_DIM,
        "encoder_frozen": False,
        "encoder_ckpt": str(ENC_CKPT.relative_to(ROOT)).replace("\\", "/"),
        "feature": FEATURE,
        "width": WIDTH,
    }
    out_ckpt = ckpt_path(agg_name, "finetune")
    out_emb = finetune_emb_path(agg_name)
    REPORTS.mkdir(exist_ok=True)
    torch.save(
        {
            "encoder": enc.state_dict(),
            "aggregator": agg.state_dict(),
            "head": head.state_dict(),
            "classes": classes,
            "config": cfg,
        },
        out_ckpt,
    )
    np.savez(out_emb, E=E, ids=ids, sess=sess, dates=dates, BG=BG)
    log(f"saved {agg_name} finetune weights BEFORE scoring -> {out_ckpt}")
    log(f"saved {agg_name} finetune embeddings BEFORE scoring -> {out_emb} E={E.shape}")
    return agg


def run_scoring():
    log(f"scoring via {RUNNER}")
    rc = subprocess.call([sys.executable, str(RUNNER)])
    if rc != 0:
        sys.exit(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", type=str, default="all",
                    help="gru|tcn|mha|all")
    ap.add_argument("--mode", type=str, default="frozen",
                    choices=["frozen", "finetune"])
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=None,
                    help="default: frozen=40, finetune=30")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--set-w", type=int, default=5)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: frozen=1e-3, finetune=5e-4")
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--scale", type=float, default=16.0)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--no-score", action="store_true",
                    help="train only; skip runner")
    args = ap.parse_args()

    if args.score_only:
        run_scoring()
        return

    if args.epochs is None:
        args.epochs = 40 if args.mode == "frozen" else 30
    if args.lr is None:
        args.lr = 1e-3 if args.mode == "frozen" else 5e-4

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("WARNING: no CUDA")
    CACHE.mkdir(parents=True, exist_ok=True)

    aggs = resolve_aggs(args.agg)
    log(f"E1b train mode={args.mode} aggs={aggs} device={device}")

    if args.mode == "frozen":
        tr, E = load_or_embed_train(device)
        for name in aggs:
            train_frozen(tr, E, name, args, device)
            if device == "cuda":
                torch.cuda.empty_cache()
    else:
        for name in aggs:
            try:
                train_finetune(name, args, device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    log(f"FINETUNE OOM on {name}: {e}")
                    log("continuing remaining aggs if any; report frozen fully")
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise
            if device == "cuda":
                torch.cuda.empty_cache()

    if not args.no_score:
        run_scoring()


if __name__ == "__main__":
    main()
