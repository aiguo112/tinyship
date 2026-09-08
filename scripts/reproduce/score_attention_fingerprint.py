#!/usr/bin/env python
"""
Re-score attention-pooled verification from SAVED encoder+attention+embeddings.
No training. GPU only needed for the tiny attention forward (CPU fine).

  python scripts/reproduce/score_attention_fingerprint.py
  python scripts/reproduce/score_attention_fingerprint.py --k-enroll 20 --probe-w 3
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.fingerprint import GatedAttentionPool  # noqa: E402

# Reuse eval + constants from the trainer module
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "taf", ROOT / "scripts" / "reproduce" / "train_attention_fingerprint.py"
)
_taf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_taf)

REPORTS = ROOT / "reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-enroll", type=int, default=20)
    ap.add_argument("--probe-w", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    embp = REPORTS / "fingerprint_attention_embeddings.npz"
    ckpt_p = REPORTS / "fingerprint_attention_ckpt.pt"
    if not embp.exists():
        sys.exit(f"missing {embp} — train first")
    if not ckpt_p.exists():
        sys.exit(f"missing {ckpt_p} — need attention weights")

    d = np.load(embp, allow_pickle=True)
    E, ids, dates = d["E"], d["ids"], d["dates"]
    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    emb_dim = ck.get("config", {}).get("emb_dim", 128)
    attn = GatedAttentionPool(emb_dim, hidden=64)
    attn.load_state_dict(ck["attention"])
    attn.eval()

    rep = _taf.evaluate_attention(
        E, ids, dates, attn, "cpu",
        k_enroll=a.k_enroll, probe_w=a.probe_w, seed=a.seed,
    )
    outp = REPORTS / "fingerprint_attention.json"
    outp.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=2, default=float))
    print(f"\nwrote {outp}")
    print(f"GATE: eer_attention <= {_taf.GATE_EER} -> pass; "
          f"free_exp floor {_taf.FREE_EXP_BASELINE}")


if __name__ == "__main__":
    main()
