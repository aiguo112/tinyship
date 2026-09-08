#!/usr/bin/env python
"""
Free experiments against intra-class spread (the diagnosed 0.24 bottleneck).
All post-processing on SAVED embeddings -- NO retrain, NO GPU.

Levers:
  1. enrollment size k     (bigger, cleaner prototype)
  2. temporal pooling w    (avg each ship's clip-embeddings in windows of w
                            consecutive same-session clips before scoring;
                            attacks per-clip jitter directly)
  3. prototype type        (mean vs medoid; medoid resists outlier clips)

DECISION GATE (pre-registered): if the BEST cell here is not clearly below the
0.24 prototype baseline (say <= ~0.20), do NOT retrain -- write the
honest-limited method paper. If it drops meaningfully, the targeted retrain
(sub-center ArcFace + temporal window) is justified.

  python scripts/reproduce/free_experiments.py --loss arcface
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def eer(pos, neg):
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    thr = np.quantile(s, np.linspace(0, 1, 1500)) if len(np.unique(s)) > 1500 else np.unique(s)
    best = (1.0, 0.5)
    for t in thr:
        p = s >= t
        far = np.mean(p[y == 0]) if (y == 0).any() else 0.0
        frr = np.mean(~p[y == 1]) if (y == 1).any() else 0.0
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return float(best[1])


def pool(E, ids, sess, w):
    """Average consecutive same-(id,session) clip embeddings in windows of w."""
    if w <= 1:
        return E, ids, sess
    nE, nid, nse = [], [], []
    order = np.lexsort((np.arange(len(ids)), sess, ids))
    Eo, ido, seo = E[order], ids[order], sess[order]
    i = 0
    while i < len(ido):
        j = i
        while j < len(ido) and ido[j] == ido[i] and seo[j] == seo[i] and (j - i) < w:
            j += 1
        v = Eo[i:j].mean(0); v /= (np.linalg.norm(v) + 1e-9)
        nE.append(v); nid.append(ido[i]); nse.append(seo[i]); i = j
    return np.array(nE), np.array(nid), np.array(nse)


def prototypes(E, gal_idx, kind):
    protos = {}
    for s, idx in gal_idx.items():
        if kind == "medoid":
            sub = E[idx]; sim = sub @ sub.T
            protos[s] = sub[sim.sum(1).argmax()]
        else:
            protos[s] = E[idx].mean(0)
        protos[s] = protos[s] / (np.linalg.norm(protos[s]) + 1e-9)
    return protos


def run(E, ids, sess, k, kind, rng):
    gal, probes = {}, []
    for s in np.unique(ids):
        idx = np.where(ids == s)[0]; rng.shuffle(idx)
        if len(idx) < k + 1:
            continue
        gal[s] = idx[:k]; probes += [(i, s) for i in idx[k:]]
    if len(gal) < 2:
        return None, 0
    protos = prototypes(E, gal, kind)
    P = np.stack(list(protos.values())); Pid = np.array(list(protos.keys()))
    pos, neg = [], []
    for i, sid in probes:
        sims = P @ E[i]
        pos.append(float(sims[Pid == sid][0])); neg.extend(sims[Pid != sid].tolist())
    return eer(np.array(pos), np.array(neg)), len(gal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="arcface")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = np.load(REPORTS / f"fingerprint_{a.loss}_embeddings.npz", allow_pickle=True)
    E = d["E"].astype(np.float64); E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ids = d["ids"].astype(str); sess = d["sess"].astype(str)

    grid = []
    for w in (1, 3, 5):
        Ew, idw, sew = pool(E, ids, sess, w)
        for k in (5, 10, 20):
            for kind in ("mean", "medoid"):
                rng = np.random.default_rng(a.seed)
                e, n = run(Ew, idw, sew, k, kind, rng)
                if e is not None:
                    grid.append({"pool_w": w, "k_enroll": k, "proto": kind,
                                 "eer": round(e, 4), "ships": n})
    grid.sort(key=lambda r: r["eer"])
    out = {"baseline_prototype_eer": 0.2401, "results": grid, "best": grid[0] if grid else None}
    (REPORTS / f"free_experiments_{a.loss}.json").write_text(json.dumps(out, indent=2))
    print(f"{'pool_w':>6} {'k':>4} {'proto':>7} {'ships':>6} {'EER':>7}")
    for r in grid:
        print(f"{r['pool_w']:>6} {r['k_enroll']:>4} {r['proto']:>7} {r['ships']:>6} {r['eer']:>7}")
    print(f"\nBEST: {out['best']}")
    print("GATE: best clearly < 0.24 (ideally <=0.20) -> retrain justified; else stop & write.")


if __name__ == "__main__":
    main()
