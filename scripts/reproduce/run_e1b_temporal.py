#!/usr/bin/env python
"""
E1b dedicated runner — score temporal aggregators on locked protocol.

Loads available reports/e1b_{agg}_{mode}_ckpt.pt checkpoints.
Frozen variants score fingerprint_attention_embeddings.npz;
finetune variants score e1b_{agg}_finetune_embeddings.npz.

Merges results into reports/e1_temporal.json under key "e1b"
without wiping e1a / e1c.

  python scripts/reproduce/run_e1b_temporal.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.certified_verify import (  # noqa: E402
    K_ENROLL, PROBE_W, aggregator_set_eval, stats_set_eval,
)
from src.models.fingerprint import build_temporal_aggregator  # noqa: E402

REPORTS = ROOT / "reports"
SEEDS = [0, 1, 2, 3, 4]
N_PROBE_REF = 7194
N_SHIPS_REF = 22
BASELINE_STATS = {"mean": 0.0629, "std": 0.0043}
TEST_EMB = REPORTS / "fingerprint_attention_embeddings.npz"
OUT_JSON = REPORTS / "e1_temporal.json"
AGGS = ("gru", "tcn", "mha")
MODES = ("frozen", "finetune")


def log(m):
    print(m, flush=True)


def _mean_std(vals):
    arr = np.array(vals, dtype=float)
    return round(float(arr.mean()), 4), round(float(arr.std(ddof=1)), 4)


def _dj(r) -> str:
    return "PASS" if r.get("disjointness_pass") else "FAIL"


def _sess_label(r) -> str:
    if "session_disjoint_pass" not in r:
        return "N/A"
    return "PASS" if r["session_disjoint_pass"] else "FAIL_SESSION"


def ckpt_path(agg: str, mode: str) -> Path:
    return REPORTS / f"e1b_{agg}_{mode}_ckpt.pt"


def emb_path(agg: str, mode: str) -> Path:
    if mode == "frozen":
        return TEST_EMB
    return REPORTS / f"e1b_{agg}_finetune_embeddings.npz"


def load_npz(path: Path):
    if not path.exists():
        sys.exit(f"BLOCKER: missing embeddings {path}")
    d = np.load(path, allow_pickle=True)
    E = d["E"]
    ids = np.asarray(d["ids"]).astype(str)
    sess = np.asarray(d["sess"]).astype(str) if "sess" in d else None
    return E, ids, sess


def load_agg(agg: str, mode: str, device: str):
    import torch

    path = ckpt_path(agg, mode)
    if not path.exists():
        return None, None
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    emb_dim = int(cfg.get("emb_dim", 128))
    name = cfg.get("agg", agg)
    mod = build_temporal_aggregator(name, emb_dim).to(device)
    mod.load_state_dict(ck["aggregator"])
    mod.eval()
    return mod, cfg


def discover_variants():
    found = []
    for mode in MODES:
        for agg in AGGS:
            if ckpt_path(agg, mode).exists():
                if mode == "finetune" and not emb_path(agg, mode).exists():
                    log(
                        f"WARNING: {ckpt_path(agg, mode).name} present but "
                        f"missing {emb_path(agg, mode).name}; skip"
                    )
                    continue
                found.append((agg, mode))
    return found


def score_variant(agg: str, mode: str, device: str) -> dict:
    mod, cfg = load_agg(agg, mode, device)
    if mod is None:
        raise FileNotFoundError(ckpt_path(agg, mode))
    E, ids, sess = load_npz(emb_path(agg, mode))

    eers, top1s = [], []
    clip_all = []
    sess_seed0 = None
    sess_labels = []
    n_ships = n_probe = None

    log(f"\n=== E1b {agg}_{mode} ===")
    for seed in SEEDS:
        r = aggregator_set_eval(
            E, ids, mod, device=device,
            k_enroll=K_ENROLL, probe_w=PROBE_W, seed=seed,
            sessions=sess,
        )
        if r["n_ships"] != N_SHIPS_REF:
            sys.exit(
                f"PROTOCOL MISMATCH: {agg}_{mode} seed={seed} "
                f"n_ships={r['n_ships']} want {N_SHIPS_REF}"
            )
        if r["n_probe"] != N_PROBE_REF:
            sys.exit(
                f"PROTOCOL MISMATCH: {agg}_{mode} seed={seed} "
                f"n_probe={r['n_probe']} want {N_PROBE_REF}"
            )
        dj = _dj(r)
        sl = _sess_label(r)
        clip_all.append(dj)
        sess_labels.append(sl)
        if dj != "PASS":
            sys.exit(f"DISJOINTNESS FAIL {agg}_{mode} seed={seed}")
        log(
            f"  seed={seed} n_ships={r['n_ships']} n_probe={r['n_probe']} "
            f"clip_disjoint={dj} session_disjoint={sl} "
            f"sess_overlap_ships={r.get('session_overlap_ships', 'n/a')} "
            f"sess_overlap_probe_frac={r.get('session_overlap_probe_frac', 'n/a')} "
            f"EER={r['eer']} top1={r['top1']}"
        )
        eers.append(r["eer"])
        top1s.append(r["top1"])
        n_ships = r["n_ships"]
        n_probe = r["n_probe"]
        if seed == 0:
            sess_seed0 = sl
            if "session_overlap_probe_frac" in r:
                sess_seed0 = (
                    f"{sl} (overlap_ships={r['session_overlap_ships']}, "
                    f"probe_frac={r['session_overlap_probe_frac']:.4f})"
                )

    eer_mean, eer_std = _mean_std(eers)
    top1_mean, top1_std = _mean_std(top1s)
    clip_status = "PASS" if all(c == "PASS" for c in clip_all) else "FAIL"
    log(
        f"  SUMMARY {agg}_{mode}: EER {eer_mean:.4f}±{eer_std:.4f} "
        f"top1 {top1_mean:.4f}±{top1_std:.4f} clip={clip_status}"
    )
    return {
        "eer_mean": eer_mean,
        "eer_std": eer_std,
        "eers": eers,
        "top1_mean": top1_mean,
        "top1_std": top1_std,
        "top1s": top1s,
        "n_ships": n_ships,
        "n_probe": n_probe,
        "clip_disjoint": clip_status,
        "session_disjoint_seed0": sess_seed0,
        "session_disjoint_seeds": sess_labels,
        "mode": mode,
        "agg": agg,
        "config": cfg,
    }


def gate_vs_stats(best_mean: float, best_std: float) -> dict:
    """Gate: mean EER <= ~0.05 AND CI non-overlapping stats-pool 0.0629±0.0043."""
    base_lo = BASELINE_STATS["mean"] - BASELINE_STATS["std"]
    base_hi = BASELINE_STATS["mean"] + BASELINE_STATS["std"]
    var_lo = best_mean - best_std
    var_hi = best_mean + best_std
    # intervals [mean-std, mean+std]; non-overlapping means var_hi < base_lo
    non_overlap = var_hi < base_lo
    mean_ok = best_mean <= 0.05
    met = bool(mean_ok and non_overlap)
    verdict = (
        f"best mean EER={best_mean:.4f}±{best_std:.4f} "
        f"[{var_lo:.4f},{var_hi:.4f}] vs stats-pool "
        f"{BASELINE_STATS['mean']:.4f}±{BASELINE_STATS['std']:.4f} "
        f"[{base_lo:.4f},{base_hi:.4f}]; "
        f"mean<=0.05={mean_ok}, CI_non_overlap={non_overlap}"
    )
    return {
        "criterion": (
            "mean EER <= ~0.05 with CI non-overlapping stats-pool 0.0629±0.0043"
        ),
        "met": met,
        "verdict": verdict,
        "stats_pool_interval": [round(base_lo, 4), round(base_hi, 4)],
        "best_interval": [round(var_lo, 4), round(var_hi, 4)],
    }


def merge_report(e1b: dict) -> Path:
    REPORTS.mkdir(exist_ok=True)
    if OUT_JSON.exists():
        rep = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    else:
        rep = {"experiment": "E1"}
    # preserve e1a skip + e1c; only replace e1b
    rep["e1b"] = e1b
    if "baseline_stats_pool" not in rep:
        rep["baseline_stats_pool"] = {
            "mean_eer": BASELINE_STATS["mean"],
            "std_eer": BASELINE_STATS["std"],
            "source": "E1c parameter-free mean+std",
        }
    OUT_JSON.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log(f"wrote {OUT_JSON}")
    return OUT_JSON


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    variants = discover_variants()
    if not variants:
        sys.exit(
            "BLOCKER: no e1b_*_*_ckpt.pt found. "
            "Train first: python scripts/reproduce/train_e1b_aggregators.py"
        )
    log(f"E1b scoring device={device} variants={variants}")

    # baseline reference (stats pool on frozen attention emb)
    if TEST_EMB.exists():
        E0, ids0, _ = load_npz(TEST_EMB)
        rs = stats_set_eval(E0, ids0, K_ENROLL, PROBE_W, 0)
        log(
            f"baseline stats_pool seed0 EER={rs['eer']} "
            f"(ref mean±std {BASELINE_STATS['mean']}±{BASELINE_STATS['std']})"
        )

    results = {}
    for agg, mode in variants:
        key = f"{agg}_{mode}"
        results[key] = score_variant(agg, mode, device)

    # pick best by eer_mean
    best_key = min(results, key=lambda k: results[k]["eer_mean"])
    best = results[best_key]
    delta = round(best["eer_mean"] - BASELINE_STATS["mean"], 4)
    gate = gate_vs_stats(best["eer_mean"], best["eer_std"])

    commands = [
        r". .\scripts\use_tinyship.ps1",
        "python scripts/reproduce/train_e1b_aggregators.py --agg all --mode frozen --no-score",
        "python scripts/reproduce/train_e1b_aggregators.py --agg all --mode finetune --no-score",
        "python scripts/reproduce/run_e1b_temporal.py",
        "python tests/test_e1b_aggregators.py",
    ]

    e1b = {
        "baseline": (
            f"e1c_stats_pool {BASELINE_STATS['mean']:.4f}±{BASELINE_STATS['std']:.4f}"
        ),
        "n_ships": N_SHIPS_REF,
        "n_probe": N_PROBE_REF,
        "variants": results,
        "best": {
            "name": best_key,
            "eer_mean": best["eer_mean"],
            "eer_std": best["eer_std"],
            "top1_mean": best["top1_mean"],
            "vs_stats_pool_delta": delta,
        },
        "gate_vs_stats_pool": gate,
        "commands": commands,
        "notes": (
            "Learned temporal aggregators (GRU/TCN/MHA) over 1s clip embeddings; "
            "frozen = agg on attention embeddings; finetune = encoder+agg. "
            "Clip-disjoint asserted (abort on FAIL); session-disjoint soft."
        ),
    }

    log(
        f"\nBEST {best_key}: EER {best['eer_mean']:.4f}±{best['eer_std']:.4f} "
        f"delta_vs_stats={delta:+.4f} gate_met={gate['met']}"
    )
    log(f"GATE: {gate['verdict']}")
    merge_report(e1b)


if __name__ == "__main__":
    main()
