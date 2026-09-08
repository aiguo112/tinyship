#!/usr/bin/env python
"""Sequential seed-variance grid: random + session × PLANNED_SEEDS.

Skips cells already in reports/reproduction_results.csv.
Population stays frozen (subset_seed=1337).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLANNED_SEEDS = (1337, 2026, 42)
SUBSET_SEED = 1337

TRAIN = ROOT / "scripts" / "reproduce" / "train_shipnn.py"
CSV = ROOT / "reports" / "reproduction_results.csv"
LOGDIR = ROOT / "reports" / "seed_runs"


def done_cells() -> set[tuple[str, int]]:
    if not CSV.exists():
        return set()
    df = pd.read_csv(CSV)
    if "seed" not in df.columns:
        df["seed"] = SUBSET_SEED
    out = set()
    for _, r in df.iterrows():
        if r.get("feature", "cqt_mfcc") != "cqt_mfcc":
            continue
        if str(r.get("aug", True)).lower() not in ("true", "1"):
            continue
        out.add((str(r["protocol"]), int(r["seed"])))
    return out


def main() -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    have = done_cells()
    jobs = [(p, s) for p in ("random", "session") for s in PLANNED_SEEDS]
    pending = [(p, s) for p, s in jobs if (p, s) not in have]
    print(f"done: {sorted(have)}")
    print(f"pending: {pending}")
    py = sys.executable
    for proto, seed in pending:
        logp = LOGDIR / f"{proto}_s{seed}.log"
        cmd = [
            py, str(TRAIN),
            "--protocol", proto,
            "--feature", "cqt_mfcc",
            "--aug",
            "--seed", str(seed),
            "--subset-seed", str(SUBSET_SEED),
        ]
        print(f"START {proto} seed={seed} -> {logp}", flush=True)
        with logp.open("a", encoding="utf-8") as f:
            f.write(f"\n# cmd: {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.call(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        if rc != 0:
            sys.exit(f"FAILED {proto} seed={seed} rc={rc}  see {logp}")
        print(f"OK {proto} seed={seed}", flush=True)
    print("seed-variance grid complete")


if __name__ == "__main__":
    main()
