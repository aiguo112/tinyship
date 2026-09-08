#!/usr/bin/env python
"""
Train TinyShip-Fingerprint on VTUAD MMSI-disjoint TRAIN ships, embed UNSEEN test
ships, save checkpoint+embeddings BEFORE scoring.

Convergence-oriented defaults (method run):
  --loss arcface --epochs 120 --margin 0.20 --scale 16 --lr 1e-3
  cosine LR decay after linear warmup

  python scripts/reproduce/train_fingerprint.py --loss arcface --dry-run
  python scripts/reproduce/train_fingerprint.py --loss arcface
  python scripts/reproduce/train_fingerprint.py --loss triplet
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
    ArcFaceHead, batch_hard_triplet, build_encoder,
)
from src.evaluation.enrollment import report as enroll_report  # noqa: E402

SPLIT_DIR = ROOT / "data" / "processed" / "splits"
CACHE = ROOT / "data" / "processed" / "feat_cache"
REPORTS = ROOT / "reports"
# Locked mmsi_*.csv: MMSI uppercase
COLS = {"path": "file_path", "mmsi": "MMSI", "session": "sub_init", "date": "date"}
FEATURE = "cqt_mfcc"
EMB_DIM, WIDTH = 128, 0.25


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_split(name):
    p = SPLIT_DIR / f"mmsi_{name}.csv"
    if not p.exists():
        sys.exit(f"missing split file {p}")
    df = pd.read_csv(p).rename(columns={v: k for k, v in COLS.items()})
    df = df[df["mmsi"].astype(str) != "0"].copy()
    df["mmsi"] = df["mmsi"].astype(str)
    return df


def load_background(name):
    p = SPLIT_DIR / f"mmsi_{name}.csv"
    df = pd.read_csv(p).rename(columns={v: k for k, v in COLS.items()})
    return df[df["mmsi"].astype(str) == "0"].copy()


class FingerprintClipDataset:
    """Module-level for Windows spawn safety (num_workers>0 later)."""

    def __init__(self, df, cmap, labeled: bool):
        self.paths = df["path"].astype(str).tolist()
        self.mmsis = df["mmsi"].astype(str).tolist()
        self.cmap = dict(cmap)
        self.labeled = bool(labeled)

    def __len__(self):
        return len(self.paths)

    def _feat(self, path: str):
        import soundfile as sf
        key = CACHE / f"{FEATURE}_{hashlib.sha256(path.encode()).hexdigest()[:16]}.npy"
        if key.exists():
            return np.load(key)
        y, sr = sf.read(path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        x = extract(y, sr or 32000, FEATURE)
        np.save(key, x)
        return x

    def __getitem__(self, i):
        import torch
        x = self._feat(self.paths[i]).copy()
        y = int(self.cmap[self.mmsis[i]]) if self.labeled else -1
        return torch.from_numpy(x), y


def _lr_at(epoch: int, epochs: int, base_lr: float, warmup: int = 5) -> float:
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    # cosine from warmup+1 .. epochs
    t = (epoch - warmup) / max(epochs - warmup, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["arcface", "triplet"], default="arcface")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.20,
                    help="ArcFace m / triplet margin (default 0.20)")
    ap.add_argument("--scale", type=float, default=16.0, help="ArcFace s (default 16)")
    ap.add_argument("--enroll-k", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    bg_te = load_background("test")
    log(f"train ships {tr['mmsi'].nunique()} clips {len(tr)} | "
        f"val ships {va['mmsi'].nunique()} | test ships {te['mmsi'].nunique()} "
        f"clips {len(te)} | test background {len(bg_te)}")

    date_counts = te.groupby("mmsi")["date"].nunique()
    log(f"test ships with >=2 dates: {(date_counts >= 2).sum()}/{te['mmsi'].nunique()} "
        f"(across_date protocol; sub_init ignored as session)")

    if a.dry_run:
        log("DRY RUN only."); return

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        log("WARNING: no CUDA")
    CACHE.mkdir(parents=True, exist_ok=True)
    nw = 0 if os.name == "nt" else 4

    classes = sorted(tr["mmsi"].unique())
    cmap = {m: i for i, m in enumerate(classes)}
    labels = np.array([cmap[m] for m in tr["mmsi"]], dtype=np.int64)
    # class-balanced sampling (helps ArcFace with uneven clip counts)
    counts = np.bincount(labels, minlength=len(classes)).astype(np.float64)
    w_per_class = 1.0 / np.maximum(counts, 1.0)
    sample_w = w_per_class[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )

    enc = build_encoder(in_channels(FEATURE), EMB_DIM, WIDTH).to(dev)
    params = list(enc.parameters())
    head = None
    if a.loss == "arcface":
        head = ArcFaceHead(EMB_DIM, len(classes), s=a.scale, m=a.margin).to(dev)
        params += list(head.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    dl = DataLoader(
        FingerprintClipDataset(tr, cmap, True),
        batch_size=a.bs,
        sampler=sampler,
        num_workers=nw,
        pin_memory=(dev == "cuda"),
        drop_last=True,
    )

    log(f"train {a.loss}: epochs={a.epochs} margin={a.margin} scale={a.scale} lr={a.lr}")
    for ep in range(1, a.epochs + 1):
        lr = _lr_at(ep, a.epochs, a.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        enc.train()
        if head is not None:
            head.train()
        running = 0.0
        n_batches = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            e = enc(x)
            if a.loss == "arcface":
                loss = torch.nn.functional.cross_entropy(head(e, y), y)
            else:
                loss = batch_hard_triplet(e, y, margin=a.margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        if ep % 10 == 0 or ep == a.epochs or ep == 1:
            log(f"epoch {ep}/{a.epochs} loss {running / max(n_batches, 1):.4f} lr={lr:.2e}")

    @torch.no_grad()
    def embed(df):
        enc.eval()
        out = []
        loader = DataLoader(
            FingerprintClipDataset(df, cmap, False),
            batch_size=256, num_workers=nw, pin_memory=(dev == "cuda"),
        )
        for x, _ in loader:
            out.append(enc(x.to(dev)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, EMB_DIM), np.float32)

    te = te.reset_index(drop=True)
    E = embed(te)
    ids = te["mmsi"].astype(str).to_numpy()
    sess = te["session"].astype(str).to_numpy()  # stored for provenance only
    dates = te["date"].astype(str).to_numpy()
    BG = embed(bg_te) if len(bg_te) else None

    REPORTS.mkdir(exist_ok=True)
    ckpt = REPORTS / f"fingerprint_{a.loss}_ckpt.pt"
    torch.save(
        {
            "encoder": enc.state_dict(),
            "classes": classes,
            "config": {
                "loss": a.loss, "epochs": a.epochs, "emb_dim": EMB_DIM,
                "margin": a.margin, "scale": a.scale, "lr": a.lr,
            },
        },
        ckpt,
    )
    embp = REPORTS / f"fingerprint_{a.loss}_embeddings.npz"
    np.savez(
        embp,
        E=E, ids=ids, sess=sess, dates=dates,
        BG=(BG if BG is not None else np.zeros((0, EMB_DIM), np.float32)),
    )
    log(f"saved weights -> {ckpt}")
    log(f"saved embeddings -> {embp}")

    try:
        rep = enroll_report(E, ids, sess, dates, BG, seed=0, enroll_k=a.enroll_k)
    except Exception as e:
        log(f"scoring failed ({e}); embeddings SAFE. "
            f"python scripts/reproduce/score_fingerprint.py --loss {a.loss}")
        raise

    rep["_config"] = {
        "loss": a.loss, "epochs": a.epochs, "emb_dim": EMB_DIM, "width": WIDTH,
        "feature": FEATURE, "margin": a.margin, "scale": a.scale, "lr": a.lr,
        "enroll_k": a.enroll_k,
    }
    outp = REPORTS / f"fingerprint_{a.loss}.json"
    outp.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    log(f"wrote {outp}")
    clip, enk, ad = rep["clip_split"], rep["enroll_k"], rep["across_date"]
    log(f"RESULT [{a.loss}] clip_split EER={clip.get('eer')} top1={clip.get('top1')} "
        f"| enroll_k={a.enroll_k} EER={enk.get('eer')} top1={enk.get('top1')} "
        f"| across_date EER={ad.get('eer')} ships={ad.get('usable_ships')} "
        f"| gap_date-clip={rep.get('eer_gap_across_date_minus_clip')}")
    log("Primary claim metrics: clip_split / enroll_k EER + top-k + OSCR. "
        "across_date is secondary (tiny N).")


if __name__ == "__main__":
    main()
