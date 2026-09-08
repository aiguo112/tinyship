#!/usr/bin/env python
"""
Multi-seed Table I: EER + top-1 over seeds {0,1,2,3,4} on existing embeddings.

  python scripts/reproduce/measure_multiseed_table1.py

No encoder training. Aborts if set-protocol rows deviate from n_ships=22, n_probe=7194.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.certified_verify import (  # noqa: E402
    asnorm_set_eval,
    attention_set_eval,
    certified_mean_eval,
    clip_cosine_eval,
    stats_set_eval,
)
from src.models.fingerprint import GatedAttentionPool  # noqa: E402

REPORTS = ROOT / "reports"
SEEDS = [0, 1, 2, 3, 4]
N_SHIPS_REF = 22
N_PROBE_REF = 7194  # set protocol only


def load_emb(name: str):
    d = np.load(REPORTS / name, allow_pickle=True)
    return d["E"], np.asarray(d["ids"]).astype(str)


def summarize(eers, top1s, meta: dict) -> dict:
    ea = np.asarray(eers, dtype=float)
    ta = np.asarray(top1s, dtype=float)
    out = {
        **meta,
        "seeds": SEEDS,
        "eers": [round(float(x), 4) for x in eers],
        "top1s": [round(float(x), 4) for x in top1s],
        "eer_mean": round(float(ea.mean()), 4),
        "eer_std": round(float(ea.std(ddof=1)), 4),
        "top1_mean": round(float(ta.mean()), 4),
        "top1_std": round(float(ta.std(ddof=1)), 4),
        "n_ships_per_seed": meta.get("n_ships_per_seed"),
        "n_probe_per_seed": meta.get("n_probe_per_seed"),
        "clip_disjoint": meta.get("clip_disjoint"),
    }
    return out


def run_seeds(name: str, fn, *, set_protocol: bool) -> dict:
    eers, top1s = [], []
    n_ships, n_probe, dj = [], [], []
    for s in SEEDS:
        r = fn(s)
        eers.append(r["eer"])
        top1s.append(r["top1"])
        n_ships.append(int(r["n_ships"]))
        n_probe.append(int(r["n_probe"]))
        dj.append(bool(r.get("disjointness_pass", False)))
        print(
            f"  [{name}] seed={s} eer={r['eer']:.4f} top1={r['top1']:.4f} "
            f"n_ships={r['n_ships']} n_probe={r['n_probe']} "
            f"disjoint={'PASS' if r.get('disjointness_pass') else 'FAIL'}"
        )
    if set_protocol:
        if any(n != N_SHIPS_REF for n in n_ships):
            raise RuntimeError(f"{name}: n_ships != {N_SHIPS_REF}: {n_ships}")
        if any(n != N_PROBE_REF for n in n_probe):
            raise RuntimeError(f"{name}: n_probe != {N_PROBE_REF}: {n_probe}")
        if not all(dj):
            raise RuntimeError(f"{name}: clip_disjoint FAIL: {dj}")
    meta = {
        "method": name,
        "set_protocol": set_protocol,
        "n_ships_per_seed": n_ships,
        "n_probe_per_seed": n_probe,
        "clip_disjoint": "PASS" if all(dj) else "FAIL",
        "protocol": None,
    }
    return summarize(eers, top1s, meta)


def main():
    import torch

    E_a, ids_a = load_emb("fingerprint_arcface_embeddings.npz")
    E_t, ids_t = load_emb("fingerprint_attention_embeddings.npz")
    E_tr, ids_tr = load_emb("fingerprint_triplet_embeddings.npz")
    E_e, ids_e = load_emb("e2_ecapa_embeddings.npz")

    ck = torch.load(
        REPORTS / "fingerprint_attention_ckpt.pt",
        map_location="cpu",
        weights_only=False,
    )
    attn = GatedAttentionPool(ck.get("config", {}).get("emb_dim", 128), hidden=64)
    attn.load_state_dict(ck["attention"])
    attn.eval()

    methods = {}

    print("=== MULTI-SEED TABLE I ===")

    # clip cosine: not set protocol (n_probe differs)
    methods["arcface_clip_cosine"] = run_seeds(
        "ArcFace + clip cosine",
        lambda s: clip_cosine_eval(E_a, ids_a, seed=s),
        set_protocol=False,
    )
    methods["arcface_clip_cosine"]["embeddings"] = "fingerprint_arcface_embeddings.npz"
    methods["arcface_clip_cosine"]["eval"] = "clip_cosine_eval"
    methods["arcface_clip_cosine"]["protocol"] = "clip_cosine"
    methods["arcface_clip_cosine"]["note"] = (
        "Not set protocol; n_probe is clip-split count (~11026), not 7194."
    )

    methods["arcface_prototype_w1"] = run_seeds(
        "ArcFace + prototype w=1",
        lambda s: certified_mean_eval(E_a, ids_a, k_enroll=20, probe_w=1, seed=s),
        set_protocol=False,
    )
    methods["arcface_prototype_w1"]["embeddings"] = "fingerprint_arcface_embeddings.npz"
    methods["arcface_prototype_w1"]["eval"] = "certified_mean_eval probe_w=1"
    methods["arcface_prototype_w1"]["protocol"] = "prototype_single_probe"
    methods["arcface_prototype_w1"]["note"] = (
        "probe_w=1 by design — NOT the set protocol (n_probe ~21600)."
    )
    # assert ships==22 for prototype
    if any(n != N_SHIPS_REF for n in methods["arcface_prototype_w1"]["n_ships_per_seed"]):
        raise RuntimeError("prototype w=1: n_ships != 22")

    methods["arcface_asnorm"] = run_seeds(
        "ArcFace + AS-norm",
        lambda s: asnorm_set_eval(E_a, ids_a, k_enroll=20, probe_w=3, seed=s),
        set_protocol=True,
    )
    methods["arcface_asnorm"]["embeddings"] = "fingerprint_arcface_embeddings.npz"
    methods["arcface_asnorm"]["eval"] = "asnorm_set_eval"
    methods["arcface_asnorm"]["protocol"] = "certified_mean_asnorm"

    methods["arcface_temporal_mean"] = run_seeds(
        "ArcFace + temporal mean",
        lambda s: certified_mean_eval(E_a, ids_a, k_enroll=20, probe_w=3, seed=s),
        set_protocol=True,
    )
    methods["arcface_temporal_mean"]["embeddings"] = "fingerprint_arcface_embeddings.npz"
    methods["arcface_temporal_mean"]["eval"] = "certified_mean_eval w=3"
    methods["arcface_temporal_mean"]["protocol"] = "certified_mean"

    methods["subcenter_attention"] = run_seeds(
        "Sub-center + attention",
        lambda s: attention_set_eval(E_t, ids_t, attn, "cpu", 20, 3, s),
        set_protocol=True,
    )
    methods["subcenter_attention"]["embeddings"] = (
        "fingerprint_attention_embeddings.npz + fingerprint_attention_ckpt.pt"
    )
    methods["subcenter_attention"]["eval"] = "attention_set_eval"
    methods["subcenter_attention"]["protocol"] = "attention_set_disjoint"

    methods["triplet_temporal_mean"] = run_seeds(
        "Triplet + temporal mean",
        lambda s: certified_mean_eval(E_tr, ids_tr, k_enroll=20, probe_w=3, seed=s),
        set_protocol=True,
    )
    methods["triplet_temporal_mean"]["embeddings"] = "fingerprint_triplet_embeddings.npz"
    methods["triplet_temporal_mean"]["eval"] = "certified_mean_eval w=3"
    methods["triplet_temporal_mean"]["protocol"] = "certified_mean"

    methods["subcenter_mean"] = run_seeds(
        "Sub-center + mean",
        lambda s: certified_mean_eval(E_t, ids_t, k_enroll=20, probe_w=3, seed=s),
        set_protocol=True,
    )
    methods["subcenter_mean"]["embeddings"] = "fingerprint_attention_embeddings.npz"
    methods["subcenter_mean"]["eval"] = "certified_mean_eval w=3"
    methods["subcenter_mean"]["protocol"] = "certified_mean"

    methods["subcenter_stats_pool"] = run_seeds(
        "Sub-center + stats-pool (ours)",
        lambda s: stats_set_eval(E_t, ids_t, k_enroll=20, probe_w=3, seed=s),
        set_protocol=True,
    )
    methods["subcenter_stats_pool"]["embeddings"] = "fingerprint_attention_embeddings.npz"
    methods["subcenter_stats_pool"]["eval"] = "stats_set_eval"
    methods["subcenter_stats_pool"]["protocol"] = "certified_stats_pool"

    methods["ecapa_stats_pool"] = run_seeds(
        "ECAPA stats-pool",
        lambda s: stats_set_eval(E_e, ids_e, k_enroll=20, probe_w=3, seed=s),
        set_protocol=True,
    )
    methods["ecapa_stats_pool"]["embeddings"] = "e2_ecapa_embeddings.npz"
    methods["ecapa_stats_pool"]["eval"] = "stats_set_eval"
    methods["ecapa_stats_pool"]["protocol"] = "certified_stats_pool"
    methods["ecapa_stats_pool"]["frontend_note"] = (
        "ECAPA-TDNN native 16 kHz fbank + AAM; pipeline-level comparison."
    )

    # Pretty table
    print("\n=== SUMMARY (mean±std) ===")
    print(f"{'Method':40} {'EER':18} {'top-1':18} {'disjoint':8}")
    for m in methods.values():
        print(
            f"{m['method']:40} "
            f"{m['eer_mean']:.4f}±{m['eer_std']:.4f}   "
            f"{m['top1_mean']:.4f}±{m['top1_std']:.4f}   "
            f"{m['clip_disjoint']}"
        )

    out = {
        "experiment": "table1_multiseed",
        "seeds": SEEDS,
        "set_protocol_ref": {
            "n_ships": N_SHIPS_REF,
            "n_probe": N_PROBE_REF,
            "k_enroll": 20,
            "probe_w": 3,
        },
        "methods": methods,
        "e1a": {
            "status": "SKIPPED_STRUCTURALLY_INFEASIBLE",
            "n_surviving_test_vessels": 2,
            "note": (
                "No comparable 22-vessel E1a EER. See reports/e1_temporal.json e1a "
                "and optional secondary_n2 in e3_oscr / e1 update."
            ),
        },
    }
    op = REPORTS / "table1_multiseed.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
