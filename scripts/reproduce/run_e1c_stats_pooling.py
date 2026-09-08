#!/usr/bin/env python
"""
E1c dedicated runner — certified mean / stats pool / ASP on locked protocol.

Loads reports/fingerprint_attention_embeddings.npz (frozen sub-center encoder).
Requires reports/e1c_asp_ckpt.pt for ASP rows (train via train_e1c_asp.py).
Writes reports/e1_temporal.json.

  python scripts/reproduce/run_e1c_stats_pooling.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.certified_verify import (  # noqa: E402
    K_ENROLL, PROBE_W, asp_set_eval, certified_mean_eval, stats_set_eval,
)
from src.models.fingerprint import AttentiveStatsPool  # noqa: E402

REPORTS = ROOT / "reports"
SEEDS = [0, 1, 2, 3, 4]
N_PROBE_REF = 7194
BASELINE_SEED0_EER = 0.0714
TEST_EMB = REPORTS / "fingerprint_attention_embeddings.npz"
ASP_CKPT = REPORTS / "e1c_asp_ckpt.pt"


def log(m):
    print(m, flush=True)


def _dj(r) -> str:
    return "PASS" if r.get("disjointness_pass") else "FAIL"


def _mean_std(eers):
    arr = np.array(eers, dtype=float)
    return round(float(arr.mean()), 4), round(float(arr.std(ddof=1)), 4)


def load_test_emb():
    if not TEST_EMB.exists():
        sys.exit(
            f"BLOCKER: missing {TEST_EMB}. Need frozen sub-center test embeddings."
        )
    d = np.load(TEST_EMB, allow_pickle=True)
    return d["E"], np.asarray(d["ids"]).astype(str)


def load_asp(device: str):
    import torch

    if not ASP_CKPT.exists():
        sys.exit(
            f"BLOCKER: missing {ASP_CKPT}. Train first: "
            "python scripts/reproduce/train_e1c_asp.py"
        )
    ck = torch.load(ASP_CKPT, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    emb_dim = int(cfg.get("emb_dim", 128))
    hidden = int(cfg.get("asp_hidden", 128))
    asp = AttentiveStatsPool(emb_dim, hidden=hidden).to(device)
    asp.load_state_dict(ck["asp"])
    asp.eval()
    return asp, cfg


def evaluate_e1c(E, ids, asp_mod, device: str) -> dict:
    commands = [
        r". .\scripts\use_tinyship.ps1",
        "python scripts/reproduce/train_e1c_asp.py",
        "python scripts/reproduce/run_e1c_stats_pooling.py",
        "python tests/test_certified_pooling.py",
    ]

    log("=== E1c certified pooling (k_enroll=20 probe_w=3 seeds=0..4) ===")
    all_dj = []

    # baseline sanity at seed 0, then all seeds
    r0 = certified_mean_eval(E, ids, K_ENROLL, PROBE_W, 0)
    log(
        f"  certified_mean seed=0 n_probe={r0['n_probe']} disjoint={_dj(r0)} "
        f"EER={r0['eer']} top1={r0['top1']}"
    )
    if r0["n_probe"] != N_PROBE_REF:
        sys.exit(
            f"PROTOCOL MISMATCH: certified_mean n_probe={r0['n_probe']} "
            f"want {N_PROBE_REF}"
        )
    if r0["eer"] != BASELINE_SEED0_EER:
        sys.exit(
            f"PROTOCOL MISMATCH: certified_mean seed=0 EER={r0['eer']} "
            f"expected {BASELINE_SEED0_EER} (sub-center + mean). STOP."
        )
    all_dj.append(_dj(r0))

    mean_eers, stats_eers, asp_eers = [], [], []
    stats_top1_s0 = asp_top1_s0 = None
    stats_dim = asp_dim = None

    for seed in SEEDS:
        rm = certified_mean_eval(E, ids, K_ENROLL, PROBE_W, seed)
        rs = stats_set_eval(E, ids, K_ENROLL, PROBE_W, seed)
        ra = asp_set_eval(E, ids, asp_mod, device, K_ENROLL, PROBE_W, seed)
        for name, r in (("certified_mean", rm), ("stats_pool", rs), ("asp", ra)):
            if r["n_probe"] != N_PROBE_REF:
                sys.exit(
                    f"PROTOCOL MISMATCH: {name} seed={seed} n_probe={r['n_probe']} "
                    f"want {N_PROBE_REF}"
                )
            dj = _dj(r)
            all_dj.append(dj)
            log(
                f"  {name:16} seed={seed} n_probe={r['n_probe']} "
                f"disjoint={dj} EER={r['eer']} top1={r['top1']}"
            )
            if dj != "PASS":
                sys.exit(f"DISJOINTNESS FAIL {name} seed={seed}")
        mean_eers.append(rm["eer"])
        stats_eers.append(rs["eer"])
        asp_eers.append(ra["eer"])
        if seed == 0:
            stats_top1_s0 = rs["top1"]
            asp_top1_s0 = ra["top1"]
            stats_dim = rs["dim"]
            asp_dim = ra["dim"]

    m_mean, m_std = _mean_std(mean_eers)
    s_mean, s_std = _mean_std(stats_eers)
    a_mean, a_std = _mean_std(asp_eers)
    overall = "PASS" if all(d == "PASS" for d in all_dj) else "FAIL"

    log(
        f"\n  baseline_subcenter_mean {m_mean:.4f} ± {m_std:.4f}  eers={mean_eers}"
    )
    log(f"  e1c_stats_pool           {s_mean:.4f} ± {s_std:.4f}  eers={stats_eers}")
    log(f"  e1c_asp                  {a_mean:.4f} ± {a_std:.4f}  eers={asp_eers}")
    log(f"  DISJOINTNESS overall: {overall}")

    return {
        "experiment": "E1c",
        "protocol": {
            "k_enroll": K_ENROLL,
            "probe_w": PROBE_W,
            "seeds": SEEDS,
            "n_probe_ref": N_PROBE_REF,
        },
        "baseline_subcenter_mean": {
            "mean": m_mean,
            "std": m_std,
            "eers": mean_eers,
            "source": "recomputed",
        },
        "e1c_stats_pool": {
            "mean": s_mean,
            "std": s_std,
            "eers": stats_eers,
            "top1_seed0": stats_top1_s0,
            "dim": stats_dim,
            "trained": False,
        },
        "e1c_asp": {
            "mean": a_mean,
            "std": a_std,
            "eers": asp_eers,
            "top1_seed0": asp_top1_s0,
            "dim": asp_dim,
            "trained": True,
        },
        "disjointness": overall,
        "commands": commands,
        "notes": "encoder frozen; only pooling changed",
    }


def write_report(rep: dict) -> Path:
    REPORTS.mkdir(exist_ok=True)
    outp = REPORTS / "e1_temporal.json"
    outp.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log(f"wrote {outp}")
    return outp


def main():
    import torch

    E, ids = load_test_emb()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asp, _cfg = load_asp(device)
    rep = evaluate_e1c(E, ids, asp, device)
    write_report(rep)


if __name__ == "__main__":
    main()
