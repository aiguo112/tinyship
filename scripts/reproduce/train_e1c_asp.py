#!/usr/bin/env python
"""
E1c — train AttentiveStatsPool on FROZEN sub-center encoder embeddings.

Encoder is never in the optimizer. Train-set clip embeddings are saved BEFORE
ASP training. Checkpoint is saved BEFORE scoring.

  python scripts/reproduce/train_e1c_asp.py
  python scripts/reproduce/train_e1c_asp.py --score-only
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
    AttentiveStatsPool, SubCenterArcFaceHead, build_encoder,
)

SPLIT_DIR = ROOT / "data" / "processed" / "splits"
CACHE = ROOT / "data" / "processed" / "feat_cache"
REPORTS = ROOT / "reports"
COLS = {"path": "file_path", "mmsi": "MMSI", "session": "sub_init", "date": "date"}
FEATURE = "cqt_mfcc"
EMB_DIM, WIDTH = 128, 0.25
ENC_CKPT = REPORTS / "fingerprint_attention_ckpt.pt"
TRAIN_EMB = REPORTS / "fingerprint_attention_train_embeddings.npz"
ASP_CKPT = REPORTS / "e1c_asp_ckpt.pt"
RUNNER = ROOT / "scripts" / "reproduce" / "run_e1c_stats_pooling.py"


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


def embed_df(df, enc, device, bs=256):
    import torch

    enc.eval()
    out = []
    plist = df["path"].astype(str).tolist()
    log(f"embedding {len(plist)} clips (frozen encoder, bs={bs})")
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


def train_asp(tr, E, args, device):
    import torch
    import torch.nn.functional as F

    ship_idx = build_ship_index(tr)
    classes = sorted(ship_idx.keys())
    cmap = {m: i for i, m in enumerate(classes)}

    mixable = sum(1 for s in classes if len(ship_idx[s]["sessions"]) >= args.set_w)
    log(
        f"train ships {len(classes)} clips {len(tr)} | set_w={args.set_w} "
        f"ships_with>={args.set_w}_sessions={mixable}/{len(classes)}"
    )

    rng_smoke = np.random.default_rng(0)
    n_sess = []
    for m in classes:
        idxs = sample_mixed_set(ship_idx[m], args.set_w, rng_smoke)
        n_sess.append(tr.loc[idxs, "session"].nunique())
    log(
        f"mixed-session sampler: mean unique sessions/set={np.mean(n_sess):.2f} "
        f"(all {len(classes)} ships, set_w={args.set_w})"
    )

    asp = AttentiveStatsPool(EMB_DIM, hidden=args.asp_hidden).to(device)
    head = SubCenterArcFaceHead(
        EMB_DIM, len(classes), K=args.K, s=args.scale, m=args.margin,
    ).to(device)
    opt = torch.optim.Adam(
        list(asp.parameters()) + list(head.parameters()), lr=args.lr,
    )
    log(
        f"train ASP only (encoder frozen, not in optimizer): epochs={args.epochs} "
        f"K={args.K} set_w={args.set_w} bs={args.bs} steps/ep={args.steps_per_epoch} "
        f"lr={args.lr} asp_hidden={args.asp_hidden}"
    )
    opt_n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    asp_n = sum(p.numel() for p in asp.parameters())
    head_n = sum(p.numel() for p in head.parameters())
    if opt_n != asp_n + head_n:
        sys.exit(f"BLOCKER: optimizer n={opt_n} != asp+head {asp_n + head_n}")

    rng = np.random.default_rng(0)
    E32 = E.astype(np.float32)
    for ep in range(1, args.epochs + 1):
        asp.train()
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
            pooled = asp(H)
            loss = F.cross_entropy(head(pooled, y), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            log(f"epoch {ep}/{args.epochs} loss {running / args.steps_per_epoch:.4f}")

    cfg = {
        "epochs": args.epochs,
        "K": args.K,
        "set_w": args.set_w,
        "margin": args.margin,
        "scale": args.scale,
        "lr": args.lr,
        "bs": args.bs,
        "steps_per_epoch": args.steps_per_epoch,
        "emb_dim": EMB_DIM,
        "asp_hidden": args.asp_hidden,
        "encoder_frozen": True,
        "encoder_ckpt": str(ENC_CKPT.relative_to(ROOT)).replace("\\", "/"),
        "feature": FEATURE,
        "width": WIDTH,
    }
    REPORTS.mkdir(exist_ok=True)
    torch.save(
        {"asp": asp.state_dict(), "head": head.state_dict(), "classes": classes, "config": cfg},
        ASP_CKPT,
    )
    log(f"saved ASP weights BEFORE scoring -> {ASP_CKPT}")
    return asp


def run_scoring():
    log(f"scoring via {RUNNER}")
    rc = subprocess.call([sys.executable, str(RUNNER)])
    if rc != 0:
        sys.exit(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true",
                    help="load existing e1c_asp_ckpt.pt + test embeddings; score only")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--set-w", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--scale", type=float, default=16.0)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--asp-hidden", type=int, default=128)
    args = ap.parse_args()

    if args.score_only:
        run_scoring()
        return

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("WARNING: no CUDA")
    CACHE.mkdir(parents=True, exist_ok=True)

    tr, E = load_or_embed_train(device)
    train_asp(tr, E, args, device)
    run_scoring()


if __name__ == "__main__":
    main()
