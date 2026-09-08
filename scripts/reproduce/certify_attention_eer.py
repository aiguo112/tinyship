#!/usr/bin/env python
"""
Independent certification of attention-eval mean EER with DISJOINTNESS ASSERTED.

Loads saved embeddings only. Rebuilds enroll/probe splits from seed=0,
asserts set(enroll) ∩ set(probe) == ∅ per ship, recomputes EER, and runs an
overlap negative control. No training, no GPU, does not modify other files.

  python scripts/reproduce/certify_attention_eer.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
K_ENROLL = 20
PROBE_W = 3
SEED = 0


def eer(pos: np.ndarray, neg: np.ndarray) -> float:
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


def mean_proto(E: np.ndarray, idx: np.ndarray) -> np.ndarray:
    v = E[idx].mean(0)
    return v / (np.linalg.norm(v) + 1e-9)


def score_all(protos: np.ndarray, Pid: np.ndarray, probes: list[tuple[np.ndarray, str]]):
    pos, neg = [], []
    for v, sid in probes:
        sims = protos @ v
        pos.append(float(sims[Pid == sid][0]))
        neg.extend(sims[Pid != sid].tolist())
    return eer(np.array(pos), np.array(neg)), len(probes)


def main():
    embp = REPORTS / "fingerprint_attention_embeddings.npz"
    if not embp.exists():
        sys.exit(f"missing {embp}")

    d = np.load(embp, allow_pickle=True)
    E = d["E"].astype(np.float64)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ids = np.asarray(d["ids"]).astype(str)
    # sess loaded for provenance / future checks; split uses seed shuffle only
    _sess = np.asarray(d["sess"]).astype(str) if "sess" in d.files else None
    del _sess

    ships = sorted(np.unique(ids).tolist())
    usable = [s for s in ships if (ids == s).sum() >= K_ENROLL + PROBE_W]
    if len(usable) < 2:
        sys.exit(f"too few usable ships: {len(usable)}")

    rng = np.random.default_rng(SEED)
    enroll_counts = {}
    protos = []
    Pid = []
    probes_disjoint: list[tuple[np.ndarray, str]] = []
    probes_overlap: list[tuple[np.ndarray, str]] = []
    disjoint_ok = True

    for s in usable:
        idx = np.where(ids == s)[0].copy()
        rng.shuffle(idx)
        enroll_idx = idx[:K_ENROLL]
        rest = idx[K_ENROLL:]
        enroll_counts[s] = int(len(enroll_idx))

        # DISJOINT probe sets from remaining only
        probe_idxs = []
        for start in range(0, len(rest) - PROBE_W + 1, PROBE_W):
            chunk = rest[start:start + PROBE_W]
            probe_idxs.extend(chunk.tolist())
            probes_disjoint.append((mean_proto(E, chunk), s))

        overlap = set(enroll_idx.tolist()) & set(probe_idxs)
        if overlap:
            print(f"DISJOINTNESS FAIL ship={s} overlap_n={len(overlap)} "
                  f"sample={sorted(list(overlap))[:5]}")
            disjoint_ok = False
            break

        protos.append(mean_proto(E, enroll_idx))
        Pid.append(s)

        # OVERLAP CONTROL: probe windows over ALL clips including enrollment
        for start in range(0, len(idx) - PROBE_W + 1, PROBE_W):
            chunk = idx[start:start + PROBE_W]
            probes_overlap.append((mean_proto(E, chunk), s))

    if not disjoint_ok:
        print("DISJOINTNESS ASSERTED: FAIL — aborting (do not trust prior EER).")
        out = {
            "certified_mean_eer": None,
            "overlap_control_eer": None,
            "n_probe_sets": 0,
            "per_ship_enroll_counts": enroll_counts,
            "disjointness_pass": False,
        }
        (REPORTS / "certify_attention_eer.json").write_text(json.dumps(out, indent=2))
        print(f"CERTIFY: mean_eer=None (json ~0.0714 expected) | "
              f"overlap_control=None | disjoint_pass=False")
        sys.exit(1)

    Pid = np.array(Pid)
    P = np.stack(protos)

    # Sanity: every ship enrolled exactly k_enroll
    assert all(c == K_ENROLL for c in enroll_counts.values()), enroll_counts

    certified, n_probe = score_all(P, Pid, probes_disjoint)
    overlap_eer, n_overlap = score_all(P, Pid, probes_overlap)

    all_k = all(c == K_ENROLL for c in enroll_counts.values())
    print(f"ships={len(usable)} enroll_clips_per_ship={K_ENROLL} (all_equal_k={all_k})")
    print(f"n_probe_sets_disjoint={n_probe} n_probe_sets_overlap_control={n_overlap}")
    print(f"certified_mean_eer={certified:.4f}")
    print(f"overlap_control_eer={overlap_eer:.4f}")
    print("DISJOINTNESS ASSERTED: PASS")

    out = {
        "protocol": "mean_proto_set_disjoint_certified",
        "k_enroll": K_ENROLL,
        "probe_w": PROBE_W,
        "seed": SEED,
        "n_ships": len(usable),
        "n_probe_sets": n_probe,
        "n_probe_sets_overlap_control": n_overlap,
        "per_ship_enroll_counts": enroll_counts,
        "enroll_clips_per_ship_all_equal_k": True,
        "certified_mean_eer": round(certified, 4),
        "overlap_control_eer": round(overlap_eer, 4),
        "expected_json_mean_ablation": 0.0714,
        "disjointness_pass": True,
    }
    outp = REPORTS / "certify_attention_eer.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {outp}")
    print(f"CERTIFY: mean_eer={certified:.4f} (json ~0.0714 expected) | "
          f"overlap_control={overlap_eer:.4f} | disjoint_pass=True")


if __name__ == "__main__":
    main()
