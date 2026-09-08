#!/usr/bin/env python
"""
Re-score fingerprint enrollment from SAVED embeddings. No training, no GPU.

  python scripts/reproduce/score_fingerprint.py --loss arcface
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.enrollment import report as enroll_report  # noqa: E402

REPORTS = ROOT / "reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["arcface", "triplet"], default="arcface")
    ap.add_argument("--enroll-k", type=int, default=5)
    a = ap.parse_args()

    embp = REPORTS / f"fingerprint_{a.loss}_embeddings.npz"
    if not embp.exists():
        sys.exit(f"no saved embeddings at {embp}. Train first.")
    d = np.load(embp, allow_pickle=True)
    E, ids = d["E"], d["ids"]
    sess = d["sess"] if "sess" in d.files else None
    dates = d["dates"] if d["dates"].size else None
    BG = d["BG"] if d["BG"].size else None

    rep = enroll_report(E, ids, sess, dates, BG, seed=0, enroll_k=a.enroll_k)
    outp = REPORTS / f"fingerprint_{a.loss}.json"
    outp.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")

    clip, enk, ad = rep["clip_split"], rep["enroll_k"], rep["across_date"]
    print(f"[{a.loss}] wrote {outp}")
    print(f"  clip_split  : EER={clip.get('eer')}  top1={clip.get('top1')}  "
          f"ships={clip.get('usable_ships')}  oscr@0.1={clip.get('ccr@fpr0.1')}")
    print(f"  enroll_k={a.enroll_k}: EER={enk.get('eer')}  top1={enk.get('top1')}  "
          f"ships={enk.get('usable_ships')}")
    print(f"  across_date : EER={ad.get('eer')}  ships={ad.get('usable_ships')}  "
          f"(secondary, tiny N)")
    print(f"  EER gap (across_date - clip_split) = "
          f"{rep.get('eer_gap_across_date_minus_clip')}")


if __name__ == "__main__":
    main()
