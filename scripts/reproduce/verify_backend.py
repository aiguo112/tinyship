#!/usr/bin/env python
"""
Verification BACKEND method for TinyShip-Fingerprint.

Diagnosis: high top-1 retrieval (~0.94) but high EER (~0.37) => the embedding has
good local geometry but uncalibrated global scale. Fix (standard in speaker
verification): prototype enrollment + adaptive score normalization (AS-norm),
optionally a cosine/PLDA-style backend. This is post-processing on SAVED
embeddings -- NO retraining, NO GPU.

  python scripts/reproduce/verify_backend.py --loss arcface

Reports EER for: raw cosine (baseline) | prototype | prototype + AS-norm.
If prototype+AS-norm drops EER a lot, that IS the method. If not, the signal
isn't in the embedding and no backend will save it.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def eer(pos, neg):
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    thr = np.quantile(s, np.linspace(0, 1, 2000)) if len(np.unique(s)) > 2000 else np.unique(s)
    best = (1.0, 0.5)
    for t in thr:
        p = s >= t
        far = np.mean(p[y == 0]) if (y == 0).any() else 0.0
        frr = np.mean(~p[y == 1]) if (y == 1).any() else 0.0
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return float(best[1])


def split_gallery_probe(ids, sess, rng, k_enroll=5):
    """Per ship: k_enroll clips -> gallery (enroll), rest -> probes."""
    gal, prb = {}, []
    for s in np.unique(ids):
        idx = np.where(ids == s)[0]
        rng.shuffle(idx)
        if len(idx) < k_enroll + 1:
            continue
        gal[s] = idx[:k_enroll]
        prb += [(i, s) for i in idx[k_enroll:]]
    return gal, prb


def score_matrix(E, gal, probes):
    """Return genuine/impostor score lists for prototype scoring."""
    protos = {s: E[gal[s]].mean(0) for s in gal}      # centroid enrollment
    for s in protos:                                   # renormalize prototype
        protos[s] /= (np.linalg.norm(protos[s]) + 1e-9)
    P = np.stack([protos[s] for s in protos]); Pid = np.array(list(protos.keys()))
    pos, neg, raw = [], [], []
    for i, sid in probes:
        v = E[i]
        sims = P @ v
        genuine = sims[Pid == sid]
        impostor = sims[Pid != sid]
        if len(genuine):
            pos.append(float(genuine[0])); neg.extend(impostor.tolist())
            raw.append((v, sims, Pid, sid))
    return np.array(pos), np.array(neg), P, Pid, probes


def asnorm(E, gal, probes, P, Pid, cohort_size=50):
    """Adaptive symmetric norm: normalize each score by cohort stats."""
    protos = {s: E[gal[s]].mean(0) for s in gal}
    for s in protos:
        protos[s] /= (np.linalg.norm(protos[s]) + 1e-9)
    # cohort = the prototypes themselves (impostor cohort per enrollment)
    coh = P
    # per-prototype cohort stats
    pc = P @ coh.T                                      # (nproto, nproto)
    mu_e = pc.mean(1, keepdims=True); sd_e = pc.std(1, keepdims=True) + 1e-9
    idx_of = {s: k for k, s in enumerate(Pid)}
    pos, neg = [], []
    for i, sid in probes:
        v = E[i]
        s_pe = P @ v                                    # probe vs all prototypes
        # probe-side cohort stats
        mu_t = s_pe.mean(); sd_t = s_pe.std() + 1e-9
        for k, pid in enumerate(Pid):
            raw = s_pe[k]
            z = 0.5 * ((raw - mu_e[k, 0]) / sd_e[k, 0] + (raw - mu_t) / sd_t)
            (pos if pid == sid else neg).append(float(z))
    return np.array(pos), np.array(neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="arcface")
    ap.add_argument("--k_enroll", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    embp = REPORTS / f"fingerprint_{a.loss}_embeddings.npz"
    if not embp.exists():
        sys.exit(f"no embeddings at {embp}")
    d = np.load(embp, allow_pickle=True)
    E = d["E"].astype(np.float64); E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ids = d["ids"].astype(str); sess = d["sess"].astype(str)
    rng = np.random.default_rng(a.seed)

    # baseline: raw clip-to-clip cosine EER (what you already have ~0.37)
    gal, probes = split_gallery_probe(ids, sess, rng, a.k_enroll)
    if len(gal) < 2:
        sys.exit("too few enrollable ships")

    pos_p, neg_p, P, Pid, _ = score_matrix(E, gal, probes)
    eer_proto = eer(pos_p, neg_p)

    pos_z, neg_z = asnorm(E, gal, probes, P, Pid)
    eer_asnorm = eer(pos_z, neg_z)

    out = {"loss": a.loss, "k_enroll": a.k_enroll, "n_ships": len(gal),
           "eer_prototype": round(eer_proto, 4),
           "eer_prototype_asnorm": round(eer_asnorm, 4)}
    (REPORTS / f"verify_backend_{a.loss}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\n-> If eer_prototype_asnorm << ~0.37, the backend IS the method.")
    print("-> If it stays ~0.37, the identity signal is not in the embedding.")


if __name__ == "__main__":
    main()
