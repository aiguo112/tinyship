#!/usr/bin/env python
"""Pack real VTUAD MMSI-test clips into a JSON blob for tsmr_demo.html."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data" / "processed" / "splits" / "mmsi_test.csv"
CACHE = ROOT / "data" / "processed" / "feat_cache"
OUT = ROOT / "tsmr_demo_samples.json"

DEMO = [
    {"mmsi": "316003289", "type": "CARGO VESSEL", "emoji": "\U0001F6A2", "zone": "B"},
    {"mmsi": "566572000", "type": "TANKER", "emoji": "\U0001F6F3", "zone": "A"},
    {"mmsi": "316001245", "type": "PASSENGER FERRY", "emoji": "\u26F4", "zone": "C"},
]
RNG = np.random.default_rng(0)


def cache_path(file_path: str) -> Path:
    h = hashlib.sha256(file_path.encode()).hexdigest()[:16]
    return CACHE / f"cqt_mfcc_{h}.npy"


def load_feat(file_path: str):
    p = cache_path(file_path)
    if not p.exists():
        return None
    x = np.load(p)
    if x.ndim != 3 or x.shape[0] < 1:
        return None
    return x.astype(np.float32)


def downsample(y: np.ndarray, n: int) -> list[float]:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return [0.0] * n
    if y.size == 1:
        return [float(y[0])] * n
    idx = np.linspace(0, y.size - 1, n)
    return np.interp(idx, np.arange(y.size), y).round(4).tolist()


def envelope(feat: np.ndarray, n: int = 160) -> list[float]:
    env = feat[0].mean(axis=0)
    env = env - env.mean()
    peak = np.max(np.abs(env)) + 1e-6
    env = env / peak
    return downsample(env, n)


def spec_strip(feat: np.ndarray, fq: int = 24, t: int = 48) -> list[list[float]]:
    cqt = feat[0]
    f_idx = np.linspace(0, cqt.shape[0] - 1, fq).astype(int)
    t_idx = np.linspace(0, cqt.shape[1] - 1, t).astype(int)
    s = cqt[f_idx][:, t_idx]
    s = (s - s.min()) / (s.max() - s.min() + 1e-6)
    return np.round(s, 3).tolist()


def cqt_vec(feat: np.ndarray) -> np.ndarray:
    v = feat[0].mean(axis=1)
    v = v - v.mean()
    n = np.linalg.norm(v) + 1e-9
    return (v / n).astype(np.float32)


def first_available(df: pd.DataFrame, k: int) -> list[tuple[str, np.ndarray]]:
    out = []
    for row in df.itertuples(index=False):
        feat = load_feat(row.file_path)
        if feat is None:
            continue
        out.append((row.file_path, feat))
        if len(out) >= k:
            break
    return out


def pca2(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    y = X @ vt[:2].T
    lo, hi = y.min(axis=0), y.max(axis=0)
    return (y - lo) / (hi - lo + 1e-9)


def main():
    df = pd.read_csv(SPLIT)
    ships = df[df["MMSI"].astype(str) != "0"].copy()
    ships["MMSI"] = ships["MMSI"].astype(str)
    bg = df[df["MMSI"].astype(str) == "0"].copy()

    stats = (
        ships.groupby("MMSI")
        .agg(
            n_clips=("file_path", "size"),
            label=("label", "first"),
            band=("distance_band", lambda s: s.mode().iloc[0]),
        )
        .reset_index()
        .sort_values("n_clips", ascending=False)
    )

    vessels = []
    for spec in DEMO:
        sub = ships[ships["MMSI"] == spec["mmsi"]]
        if sub.empty:
            raise SystemExit(f"missing MMSI {spec['mmsi']}")
        sub = sub.sample(frac=1, random_state=0)
        packed = first_available(sub, 25)
        if len(packed) < 8:
            raise SystemExit(f"not enough cached clips for {spec['mmsi']}: {len(packed)}")
        enroll = [cqt_vec(f) for _, f in packed[:20]]
        proto = np.mean(enroll, axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-9)
        probes = packed[20:25] if len(packed) >= 25 else packed[15:20]
        probe_block = []
        for path, feat in probes:
            row = sub[sub["file_path"] == path].iloc[0]
            vec = cqt_vec(feat)
            cos = float(np.dot(proto, vec))
            probe_block.append(
                {
                    "file": Path(path).name,
                    "band": str(row["distance_band"]),
                    "cosine": round(cos, 4),
                    "match_pct": round(max(0.0, min(99.9, (cos + 1) / 2 * 100)), 1),
                    "wave": envelope(feat),
                    "spec": spec_strip(feat),
                }
            )
        label = str(sub["label"].iloc[0])
        band = str(stats.loc[stats["MMSI"] == spec["mmsi"], "band"].iloc[0])
        n_clips = int(stats.loc[stats["MMSI"] == spec["mmsi"], "n_clips"].iloc[0])
        vessels.append(
            {
                **spec,
                "label": label,
                "band": band,
                "n_clips": n_clips,
                "n_sessions": int(sub["sub_init"].nunique()),
                "probes": probe_block,
            }
        )

    # ~300-point CQT PCA (encoder .npz is not in the tree)
    vecs, meta = [], []
    per_ship = 12
    for mmsi, g in ships.groupby("MMSI"):
        g = g.sample(n=min(len(g), 80), random_state=0)
        got = 0
        for row in g.itertuples(index=False):
            feat = load_feat(row.file_path)
            if feat is None:
                continue
            vecs.append(cqt_vec(feat))
            meta.append({"mmsi": str(mmsi), "label": str(row.label), "bg": False})
            got += 1
            if got >= per_ship:
                break

    bg_s = bg.sample(n=min(len(bg), 200), random_state=0) if len(bg) else bg
    n_bg = 0
    for row in bg_s.itertuples(index=False):
        feat = load_feat(row.file_path)
        if feat is None:
            continue
        vecs.append(cqt_vec(feat))
        meta.append({"mmsi": "0", "label": "background", "bg": True})
        n_bg += 1
        if n_bg >= 50:
            break

    X = np.stack(vecs)
    try:
        from sklearn.manifold import TSNE

        xy = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=0,
        ).fit_transform(X)
        xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-9)
        method = "t-SNE of mean-pooled CQT (95-d)"
    except Exception as e:
        xy = pca2(X)
        method = f"PCA of mean-pooled CQT (95-d); t-SNE unavailable ({type(e).__name__})"

    points = []
    for (x, y), m in zip(xy, meta):
        points.append(
            {
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "mmsi": m["mmsi"],
                "label": m["label"],
                "bg": m["bg"],
            }
        )

    payload = {
        "source": "VTUAD MMSI-disjoint test split + CQT+MFCC feat_cache",
        "note": (
            "Waveforms and scatter use real 1 s clip features. "
            "128-d encoder weights/embeddings are not in the repo, so match scores "
            "are cosine vs a k=20 mean-pooled CQT prototype (not ArcFace)."
        ),
        "protocol": {
            "n_unseen_ships": 22,
            "n_test_clips": int(len(df)),
            "n_ship_clips": int(len(ships)),
            "n_background": int(len(bg)),
            "emb_dim": 128,
            "eer_stats_pool": 0.063,
            "top1_stats_pool": 0.908,
            "k_enroll": 20,
            "probe_w": 3,
            "scatter": method,
        },
        "ships": stats.to_dict(orient="records"),
        "vessels": vessels,
        "points": points,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print("vessels", [(v["mmsi"], v["type"], v["band"], len(v["probes"])) for v in vessels])
    print("points", len(points), "bg", sum(p["bg"] for p in points))


if __name__ == "__main__":
    main()
