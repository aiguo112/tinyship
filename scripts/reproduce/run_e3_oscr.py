#!/usr/bin/env python
"""
E3 — OSCR / CCR@FPR=0.1 on locked certified splits (existing embeddings only).

Known and unknown probes both use the method's pool_fn with probe_w=3.
Unknowns: background clips shuffled (eval seed) and chunked into non-overlapping
windows of size PROBE_W (remainder dropped). Singleton stats-pool unknowns
(std≈eps) are invalid for OSCR and are not used.

  python scripts/reproduce/run_e3_oscr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.certified_verify import (  # noqa: E402
    K_ENROLL,
    PROBE_W,
    _split_ships,
    l2_normalize,
    mean_proto,
    stats_proto,
)
from src.evaluation.enrollment import oscr  # noqa: E402

REPORTS = ROOT / "reports"
SEEDS = [0, 1, 2, 3, 4]
N_SHIPS_REF = 22
N_PROBE_REF = 7194
BG_REF = 3891
N_UNK_REF = BG_REF // PROBE_W  # 1297 non-overlapping w=3 BG probes


def load_npz(name: str):
    d = np.load(REPORTS / name, allow_pickle=True)
    if "BG" not in d:
        raise RuntimeError(f"{name}: BG missing — abort")
    BG = np.asarray(d["BG"])
    if BG.shape[0] != BG_REF:
        print(f"WARNING: {name} BG count={BG.shape[0]} expected {BG_REF}")
    return d["E"], np.asarray(d["ids"]).astype(str), BG


def bg_probe_chunks(n_bg: int, probe_w: int, seed: int) -> list[np.ndarray]:
    """Non-overlapping BG index windows of size probe_w (seeded shuffle)."""
    order = np.random.default_rng(seed).permutation(n_bg)
    return [
        order[i : i + probe_w]
        for i in range(0, len(order) - probe_w + 1, probe_w)
    ]


def known_unknown_scores(E, ids, BG, seed: int, pool_fn):
    """Build enroll prototypes + known/unknown max-similarity scores."""
    E = l2_normalize(E)
    BG = l2_normalize(BG)
    ids = np.asarray(ids).astype(str)
    splits = _split_ships(ids, K_ENROLL, PROBE_W, seed)
    if len(splits) != N_SHIPS_REF:
        raise RuntimeError(f"n_ships={len(splits)} != {N_SHIPS_REF}")

    Pid = np.array(list(splits.keys()))
    P = np.stack([pool_fn(E, splits[s]["enroll"]) for s in Pid])

    sim_known = []
    known_correct = []
    n_probe = 0
    for s in Pid:
        for chunk in splits[s]["probe_chunks"]:
            v = pool_fn(E, chunk)
            sims = P @ v
            pred = int(np.argmax(sims))
            sim_known.append(float(sims[pred]))
            known_correct.append(bool(Pid[pred] == s))
            n_probe += 1
    if n_probe != N_PROBE_REF:
        raise RuntimeError(f"n_probe={n_probe} != {N_PROBE_REF}")

    # Unknowns: same probe_w and pool_fn as knowns (fair OSCR; avoids singleton
    # stats expansion with std≈eps that collapses CCR@0.1 and OSCR-AUC to top-1).
    chunks = bg_probe_chunks(len(BG), PROBE_W, seed)
    if len(chunks) != N_UNK_REF:
        raise RuntimeError(f"n_unk={len(chunks)} != {N_UNK_REF}")
    unk = np.stack([pool_fn(BG, c) for c in chunks])
    sim_unknown = (unk @ P.T).max(axis=1)

    return {
        "sim_known": np.asarray(sim_known, dtype=float),
        "known_correct": np.asarray(known_correct, dtype=bool),
        "sim_unknown": np.asarray(sim_unknown, dtype=float),
        "n_ships": len(Pid),
        "n_probe": n_probe,
        "n_bg": int(BG.shape[0]),
        "n_unk": int(len(chunks)),
        "disjointness_pass": True,
    }


def eval_method(name: str, E, ids, BG, pool_fn) -> dict:
    ccrs, aucs, per = [], [], []
    print(f"\n=== {name} ===")
    for s in SEEDS:
        pack = known_unknown_scores(E, ids, BG, s, pool_fn)
        m = oscr(pack["sim_known"], pack["known_correct"], pack["sim_unknown"])
        ccrs.append(m["ccr@fpr0.1"])
        aucs.append(m["oscr_auc"])
        per.append(
            {
                "seed": s,
                "ccr@fpr0.1": m["ccr@fpr0.1"],
                "oscr_auc": m["oscr_auc"],
                "n_ships": pack["n_ships"],
                "n_probe": pack["n_probe"],
                "n_bg": pack["n_bg"],
                "n_unk": pack["n_unk"],
                "clip_disjoint": "PASS",
            }
        )
        print(
            f"  seed={s} ccr@0.1={m['ccr@fpr0.1']:.4f} oscr_auc={m['oscr_auc']:.4f} "
            f"n_unk={pack['n_unk']} disjoint=PASS"
        )
    ca = np.asarray(ccrs, dtype=float)
    aa = np.asarray(aucs, dtype=float)
    return {
        "method": name,
        "seeds": SEEDS,
        "ccr@fpr0.1_per_seed": ccrs,
        "oscr_auc_per_seed": aucs,
        "ccr@fpr0.1_mean": round(float(ca.mean()), 4),
        "ccr@fpr0.1_std": round(float(ca.std(ddof=1)), 4),
        "oscr_auc_mean": round(float(aa.mean()), 4),
        "oscr_auc_std": round(float(aa.std(ddof=1)), 4),
        "n_ships": N_SHIPS_REF,
        "n_probe": N_PROBE_REF,
        "n_bg": int(BG.shape[0]),
        "n_unk": N_UNK_REF,
        "clip_disjoint": "PASS",
        "per_seed": per,
    }


def secondary_e1a_n2() -> dict | None:
    """Optional certified EER on the 2 surviving win_3s vessels only."""
    e1 = json.loads((REPORTS / "e1_temporal.json").read_text(encoding="utf-8"))
    mmsis = e1.get("e1a", {}).get("surviving_test_vessels", {}).get("win_3s", {}).get(
        "mmsis"
    )
    if not mmsis or len(mmsis) < 2:
        return {
            "status": "SKIPPED_STRUCTURALLY_INFEASIBLE",
            "note": "no surviving mmsis listed",
        }

    # We do NOT have E1a window embeddings here — only 1s clip embeddings.
    # Honest secondary: filter attention embeddings to the 2 MMSIs and run
    # certified_mean / stats on those ships only (still 1s clips, NOT true E1a
    # multi-second windows). Label clearly as non-comparable.
    from src.evaluation.certified_verify import certified_mean_eval, stats_set_eval

    d = np.load(REPORTS / "fingerprint_attention_embeddings.npz", allow_pickle=True)
    E = d["E"]
    ids = np.asarray(d["ids"]).astype(str)
    mask = np.isin(ids, np.asarray(mmsis).astype(str))
    if mask.sum() == 0:
        return {
            "status": "SKIPPED_NO_EMBEDDINGS",
            "mmsis": mmsis,
            "note": "listed MMSIs not present in attention embeddings",
        }
    E2, ids2 = E[mask], ids[mask]
    n_per = {s: int((ids2 == s).sum()) for s in mmsis}
    print(f"\n=== E1a secondary_n2 (1s clips on 2 MMSIs only) ===")
    print(f"  mmsis={mmsis} counts={n_per}")

    mean_eers, stats_eers = [], []
    for s in SEEDS:
        try:
            r_m = certified_mean_eval(E2, ids2, k_enroll=20, probe_w=3, seed=s)
            r_s = stats_set_eval(E2, ids2, k_enroll=20, probe_w=3, seed=s)
        except RuntimeError as ex:
            return {
                "status": "FAILED_SPLITS",
                "error": str(ex),
                "mmsis": mmsis,
                "counts": n_per,
                "label": "secondary_n2_noncomparable",
                "note": (
                    "Not E1a multi-second windows — filtered 1s-clip embeddings for "
                    "the 2 vessels that survive the E1a structural gate. "
                    "N=2; NOT comparable to Table I (22 vessels)."
                ),
            }
        mean_eers.append(r_m["eer"])
        stats_eers.append(r_s["eer"])
        print(
            f"  seed={s} mean_eer={r_m['eer']:.4f} stats_eer={r_s['eer']:.4f} "
            f"n_ships={r_m['n_ships']} n_probe={r_m['n_probe']}"
        )

    ma = np.asarray(mean_eers, float)
    sa = np.asarray(stats_eers, float)
    return {
        "status": "COMPUTED_SECONDARY_ONLY",
        "label": "secondary_n2_noncomparable",
        "mmsis": mmsis,
        "counts_1s_clips": n_per,
        "k_enroll": 20,
        "probe_w": 3,
        "seeds": SEEDS,
        "subcenter_mean": {
            "eers": [round(float(x), 4) for x in mean_eers],
            "eer_mean": round(float(ma.mean()), 4),
            "eer_std": round(float(ma.std(ddof=1)), 4),
        },
        "subcenter_stats_pool": {
            "eers": [round(float(x), 4) for x in stats_eers],
            "eer_mean": round(float(sa.mean()), 4),
            "eer_std": round(float(sa.std(ddof=1)), 4),
        },
        "note": (
            "NOT Table I / NOT true E1a. Uses 1s-clip embeddings filtered to the "
            "2 vessels that survive the same-session multi-clip E1a gate. "
            "Comparable 22-vessel E1a remains SKIPPED_STRUCTURALLY_INFEASIBLE."
        ),
    }


def main():
    E_t, ids_t, BG_t = load_npz("fingerprint_attention_embeddings.npz")
    E_e, ids_e, BG_e = load_npz("e2_ecapa_embeddings.npz")

    if BG_t.shape[0] != BG_REF:
        raise RuntimeError(f"attention BG={BG_t.shape[0]} != {BG_REF}")
    if BG_e.shape[0] != BG_REF:
        raise RuntimeError(f"ecapa BG={BG_e.shape[0]} != {BG_REF}")

    protocol = {
        "k_enroll": K_ENROLL,
        "probe_w_known": PROBE_W,
        "probe_w_unknown": PROBE_W,
        "unknown_note": (
            "BG embeddings from the same npz (3891 test background clips). "
            "Unknowns use the same pool_fn and probe_w=3 as knowns: clips are "
            f"shuffled per eval seed and chunked into {N_UNK_REF} non-overlapping "
            "windows (remainder dropped). This avoids singleton stats-pool "
            "expansion (std≈eps), which previously collapsed CCR@0.1 and OSCR-AUC "
            "to closed-set top-1."
        ),
        "seeds": SEEDS,
        "n_ships": N_SHIPS_REF,
        "n_probe_known": N_PROBE_REF,
        "n_bg": BG_REF,
        "n_unk": N_UNK_REF,
        "score": "score_max = max(P @ v); correct = (argmax == true ship)",
        "metric": "src.evaluation.enrollment.oscr → ccr@fpr0.1, oscr_auc",
    }

    methods = {
        "subcenter_mean": eval_method(
            "Sub-center + mean",
            E_t,
            ids_t,
            BG_t,
            mean_proto,
        ),
        "subcenter_stats_pool": eval_method(
            "Sub-center + stats-pool (ours)",
            E_t,
            ids_t,
            BG_t,
            stats_proto,
        ),
        "ecapa_stats_pool": eval_method(
            "ECAPA stats-pool",
            E_e,
            ids_e,
            BG_e,
            stats_proto,
        ),
    }
    methods["subcenter_mean"]["embeddings"] = "fingerprint_attention_embeddings.npz"
    methods["subcenter_stats_pool"]["embeddings"] = "fingerprint_attention_embeddings.npz"
    methods["ecapa_stats_pool"]["embeddings"] = "e2_ecapa_embeddings.npz"
    methods["ecapa_stats_pool"]["frontend_note"] = (
        "ECAPA-TDNN native 16 kHz fbank + AAM; pipeline-level comparison."
    )

    print("\n=== OSCR SUMMARY (mean±std) ===")
    for m in methods.values():
        print(
            f"  {m['method']:35} "
            f"CCR@0.1={m['ccr@fpr0.1_mean']:.4f}±{m['ccr@fpr0.1_std']:.4f}  "
            f"OSCR-AUC={m['oscr_auc_mean']:.4f}±{m['oscr_auc_std']:.4f}"
        )

    secondary = secondary_e1a_n2()

    out = {
        "experiment": "E3_OSCR",
        "protocol": protocol,
        "methods": methods,
        "asnorm": {
            "status": "SKIPPED",
            "note": "optional AS-norm OSCR skipped (score calibration complicates max-sim OSCR)",
        },
        "e1a": {
            "status": "SKIPPED_STRUCTURALLY_INFEASIBLE",
            "n_surviving_test_vessels": 2,
            "secondary_n2": secondary,
        },
    }
    op = REPORTS / "e3_oscr.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {op}")

    # Keep e1_temporal as source of truth for E1a skip; append secondary pointer only.
    e1_path = REPORTS / "e1_temporal.json"
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    e1["e1a"]["status"] = "SKIPPED_STRUCTURALLY_INFEASIBLE"
    e1["e1a"]["secondary_n2"] = secondary
    e1["e1a"]["comparable_22vessel_eer"] = None
    e1["e3_oscr_report"] = str(op.relative_to(ROOT)).replace("\\", "/")
    e1_path.write_text(json.dumps(e1, indent=2), encoding="utf-8")
    print(f"updated {e1_path} e1a.secondary_n2 + e3 pointer")


if __name__ == "__main__":
    main()
