#!/usr/bin/env python
"""
MMSI-level (vessel) bootstrap 95% CIs for primary set-protocol methods.

Units of resampling = evaluation vessels, not clips and not only seeds.

  python scripts/reproduce/measure_mmsi_bootstrap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.certified_verify import (  # noqa: E402
    _pack_set_protocol,
    hierarchical_mmsi_bootstrap,
    mean_proto,
    mmsi_bootstrap_ci,
    stats_proto,
)

REPORTS = ROOT / "reports"
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 1000
SEED0_BOOT = 1000


def load_emb(name: str):
    d = np.load(REPORTS / name, allow_pickle=True)
    return d["E"], np.asarray(d["ids"]).astype(str)


def packs_for(E, ids, pool_fn):
    packs = []
    for s in SEEDS:
        Pid, P, V, yi = _pack_set_protocol(E, ids, pool_fn, seed=s)
        sim = V @ P.T
        packs.append({"seed": s, "Pid": Pid, "P": P, "V": V, "yi": yi, "sim": sim})
        print(f"  packed seed={s} n_ships={len(Pid)} n_probe={len(yi)} dim={P.shape[1]}")
    return packs


def summarize_method(name: str, packs: list[dict]) -> dict:
    # Seed-0 vessel bootstrap (fixed split) + hierarchical (seed then vessel).
    pack0 = next(p for p in packs if p["seed"] == 0)
    ci0 = mmsi_bootstrap_ci(
        pack0["P"], pack0["yi"], sim=pack0["sim"], n_boot=SEED0_BOOT, seed=1000
    )
    print(
        f"  [{name}] seed=0 "
        f"EER CI95={ci0['eer_ci95']} top1 CI95={ci0['top1_ci95']} "
        f"mean_unique_ships={ci0['mean_unique_ships']}"
    )
    hier = hierarchical_mmsi_bootstrap(packs, n_boot=N_BOOT, seed=42)
    print(
        f"  [{name}] hierarchical "
        f"EER CI95={hier['eer_ci95']} top1 CI95={hier['top1_ci95']}"
    )
    return {
        "method": name,
        "n_boot_hierarchical": N_BOOT,
        "n_boot_seed0": SEED0_BOOT,
        "seed0_mmsi_bootstrap": ci0,
        "hierarchical_seed_then_mmsi": hier,
        "primary_ci": hier,
    }


def main():
    print("=== MMSI BOOTSTRAP CIs ===")
    E_t, ids_t = load_emb("fingerprint_attention_embeddings.npz")
    E_e, ids_e = load_emb("e2_ecapa_embeddings.npz")

    out = {
        "experiment": "mmsi_bootstrap_ci",
        "n_boot": N_BOOT,
        "alpha": 0.05,
        "note": (
            "Independent units = evaluation MMSIs. Each bootstrap resamples 22 vessels "
            "with replacement, keeps the unique set, and recomputes EER/top-1. "
            "Hierarchical variant first draws an eval-seed pack (0..4), then resamples MMSIs."
        ),
        "methods": {},
    }

    print("\n--- Sub-center + mean ---")
    out["methods"]["subcenter_mean"] = summarize_method(
        "Sub-center + mean", packs_for(E_t, ids_t, mean_proto)
    )
    print("\n--- Sub-center + stats-pool (ours) ---")
    out["methods"]["subcenter_stats_pool"] = summarize_method(
        "Sub-center + stats-pool (ours)", packs_for(E_t, ids_t, stats_proto)
    )
    print("\n--- ECAPA stats-pool ---")
    out["methods"]["ecapa_stats_pool"] = summarize_method(
        "ECAPA stats-pool", packs_for(E_e, ids_e, stats_proto)
    )

    op = REPORTS / "mmsi_bootstrap.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {op}")

    # Attach primary CI to table1 report for paper sync.
    t1_path = REPORTS / "table1_multiseed.json"
    t1 = json.loads(t1_path.read_text(encoding="utf-8"))
    for key, block in out["methods"].items():
        if key in t1.get("methods", {}):
            t1["methods"][key]["eer_ci95_mmsi"] = block["primary_ci"]["eer_ci95"]
            t1["methods"][key]["top1_ci95_mmsi"] = block["primary_ci"]["top1_ci95"]
            t1["methods"][key]["ci_note"] = (
                "95% percentile CI from hierarchical bootstrap "
                "(draw eval seed, then resample MMSIs); see reports/mmsi_bootstrap.json"
            )
    t1["mmsi_bootstrap_report"] = str(op.relative_to(ROOT)).replace("\\", "/")
    t1_path.write_text(json.dumps(t1, indent=2), encoding="utf-8")
    print(f"updated {t1_path}")


if __name__ == "__main__":
    main()
