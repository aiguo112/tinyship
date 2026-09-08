#!/usr/bin/env python
"""
E2.1 score-only — reload reports/e2_ecapa_embeddings.npz and run certified eval.

  python scripts/reproduce/score_e2_ecapa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.reproduce.train_e2_ecapa import run_score_only, log  # noqa: E402


def main():
    e2_1 = run_score_only()
    mp, sp = e2_1["mean_pool"], e2_1["stats_pool"]
    log(
        f"RESULT mean_pool EER={mp['eer_mean']}±{mp['eer_std']} "
        f"top1={mp['top1_mean']}±{mp['top1_std']}"
    )
    log(
        f"RESULT stats_pool EER={sp['eer_mean']}±{sp['eer_std']} "
        f"top1={sp['top1_mean']}±{sp['top1_std']}"
    )
    log(f"clip_disjoint={e2_1['clip_disjoint']} n_ships={e2_1['n_ships']} "
        f"n_probe={e2_1['n_probe']}")


if __name__ == "__main__":
    main()
