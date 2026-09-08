#!/usr/bin/env python
"""
PART A — Seed variance + Table I protocol certification.

  python scripts/reproduce/table1_seedvariance.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.certified_verify import (  # noqa: E402
    K_ENROLL, PROBE_W,
    asnorm_set_eval, attention_set_eval, certified_mean_eval, clip_cosine_eval,
)
from src.models.fingerprint import GatedAttentionPool  # noqa: E402

REPORTS = ROOT / "reports"
SEEDS = [0, 1, 2, 3, 4]


def load_emb(name):
    d = np.load(REPORTS / name, allow_pickle=True)
    return d["E"], np.asarray(d["ids"]).astype(str)


def main():
    import torch

    E_a, ids_a = load_emb("fingerprint_arcface_embeddings.npz")
    E_t, ids_t = load_emb("fingerprint_attention_embeddings.npz")
    trip_path = REPORTS / "fingerprint_triplet_embeddings.npz"
    E_tr = ids_tr = None
    if trip_path.exists():
        E_tr, ids_tr = load_emb("fingerprint_triplet_embeddings.npz")

    ck = torch.load(REPORTS / "fingerprint_attention_ckpt.pt", map_location="cpu", weights_only=False)
    attn = GatedAttentionPool(ck.get("config", {}).get("emb_dim", 128), hidden=64)
    attn.load_state_dict(ck["attention"])
    attn.eval()

    # --- Protocol check for Table I rows at seed=0 ---
    print("=== TABLE I PROTOCOL CHECK (seed=0) ===")
    protocol_rows = []

    def add_row(method, r, k, w, seed, note=""):
        n = r.get("n_probe") or r.get("n_probe_sets")
        dj = "PASS" if r.get("disjointness_pass") else "FAIL"
        protocol_rows.append({
            "method": method, "k_enroll": k, "probe_w": w, "seed": seed,
            "n_probe_sets": n, "disjoint": dj, "eer": r.get("eer"), "top1": r.get("top1"),
            "note": note, "protocol": r.get("protocol"),
        })
        print(f"  {method:55} | k={k} w={w} seed={seed} n_probe={n} disjoint={dj} EER={r.get('eer')}")

    # Closed-set: N/A
    protocol_rows.append({
        "method": "Closed-set (ShipNN-style softmax)", "k_enroll": None, "probe_w": None,
        "seed": None, "n_probe_sets": None, "disjoint": "N/A", "eer": None, "top1": None,
        "note": "undefined on unseen",
    })
    print(f"  {'Closed-set':55} | undefined")

    r = clip_cosine_eval(E_a, ids_a, seed=0)
    add_row("ArcFace + clip cosine", r, "N/A", "N/A", 0, "clip-level backend")

    r = certified_mean_eval(E_a, ids_a, k_enroll=20, probe_w=1, seed=0)
    add_row("ArcFace + prototype enroll (single-clip probe)", r, 20, 1, 0,
            "probe_w=1 by design — NOT the set protocol")

    r = asnorm_set_eval(E_a, ids_a, k_enroll=20, probe_w=3, seed=0)
    add_row("ArcFace + AS-norm", r, 20, 3, 0)

    r = certified_mean_eval(E_a, ids_a, k_enroll=20, probe_w=3, seed=0)
    add_row("ArcFace + temporal mean", r, 20, 3, 0)
    n_ref = r["n_probe"]

    r = attention_set_eval(E_t, ids_t, attn, "cpu", 20, 3, 0)
    add_row("Sub-center + attention", r, 20, 3, 0)

    if E_tr is not None:
        r = certified_mean_eval(E_tr, ids_tr, 20, 3, 0)
        add_row("Triplet + temporal mean", r, 20, 3, 0)

    r = certified_mean_eval(E_t, ids_t, 20, 3, 0)
    add_row("Sub-center + mean (ours)", r, 20, 3, 0)

    # Check set-protocol rows share k=20,w=3,seed=0,same n_probe, PASS
    set_rows = [p for p in protocol_rows
                if p["k_enroll"] == 20 and p["probe_w"] == 3 and p["seed"] == 0]
    fail = False
    for p in set_rows:
        if p["disjoint"] != "PASS":
            print(f"PROTOCOL BUG: disjoint FAIL for {p['method']}")
            fail = True
        if p["n_probe_sets"] != n_ref:
            print(f"PROTOCOL BUG: n_probe mismatch {p['method']} got {p['n_probe_sets']} want {n_ref}")
            fail = True
    if fail:
        print("ABORT: fix protocol before paper edits")
        sys.exit(1)
    print(f"SET-PROTOCOL ROWS OK: k=20 w=3 seed=0 n_probe={n_ref} disjoint=PASS ({len(set_rows)} rows)")

    print("\n=== AS-NORM 0.081 vs earlier ~0.24 ===")
    print("Earlier verify_backend: k_enroll=5, single-clip probes, raw AS-norm on ArcFace emb → EER~0.24")
    print("Table I AS-norm:       k_enroll=20, probe_w=3 mean-pooled probes, same AS-norm → EER~0.081")
    print("Difference: enrollment size + temporal probe pooling (NOT a different AS-norm formula).")

    # --- Seed variance for top 3 ---
    print("\n=== SEED VARIANCE (seeds 0..4) ===")
    top = {
        "ours_mean": [],
        "attention": [],
        "asnorm": [],
    }
    per_seed = []
    for s in SEEDS:
        o = certified_mean_eval(E_t, ids_t, 20, 3, s)
        a = attention_set_eval(E_t, ids_t, attn, "cpu", 20, 3, s)
        z = asnorm_set_eval(E_a, ids_a, 20, 3, s)
        top["ours_mean"].append(o["eer"])
        top["attention"].append(a["eer"])
        top["asnorm"].append(z["eer"])
        per_seed.append({"seed": s, "ours_mean": o["eer"], "attention": a["eer"], "asnorm": z["eer"],
                         "n_probe_ours": o["n_probe"], "disjoint_ours": o["disjointness_pass"]})
        print(f"  seed={s}: ours={o['eer']:.4f} attn={a['eer']:.4f} asnorm={z['eer']:.4f} "
              f"n_probe={o['n_probe']} disjoint={o['disjointness_pass']}")

    stats = {}
    for k, vals in top.items():
        arr = np.array(vals, dtype=float)
        stats[k] = {
            "eers": vals,
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
        }
        print(f"  {k}: {stats[k]['mean']:.4f} ± {stats[k]['std']:.4f}  "
              f"(range {stats[k]['min']:.4f}–{stats[k]['max']:.4f})")

    # Overlap of mean±std intervals (simple CI: mean±std)
    def interval(st):
        return (st["mean"] - st["std"], st["mean"] + st["std"])

    io, ia, iz = interval(stats["ours_mean"]), interval(stats["attention"]), interval(stats["asnorm"])
    # Tie if intervals overlap pairwise among top
    def overlaps(a, b):
        return not (a[1] < b[0] or b[1] < a[0])

    tied = overlaps(io, ia) or overlaps(io, iz) or overlaps(ia, iz)
    # More strict: if ours interval overlaps BOTH others → TOP METHODS TIED
    # Ranking significant only if ours is strictly better (upper ours < lower of next best)
    next_best = min(stats["attention"]["mean"], stats["asnorm"]["mean"])
    next_std = stats["attention"]["std"] if stats["attention"]["mean"] <= stats["asnorm"]["mean"] else stats["asnorm"]["std"]
    ours_upper = stats["ours_mean"]["mean"] + stats["ours_mean"]["std"]
    next_lower = next_best - next_std
    ranking_sig = ours_upper < next_lower
    verdict = "RANKING SIGNIFICANT" if ranking_sig else "TOP METHODS TIED"

    print(f"\nIntervals: ours={io} attn={ia} asnorm={iz}")
    print(f"ours_upper={ours_upper:.4f} next_lower={next_lower:.4f}")
    print(f"VERDICT: {verdict}")

    out = {
        "protocol_check": protocol_rows,
        "set_protocol_n_probe_ref": n_ref,
        "asnorm_explanation": (
            "verify_backend used k_enroll=5 + single-clip probes (EER~0.24); "
            "Table I uses k_enroll=20 + probe_w=3 mean-pooled probes (EER~0.081). "
            "Same AS-norm formula; different enrollment/probe aggregation."
        ),
        "seed_variance": stats,
        "per_seed": per_seed,
        "verdict": verdict,
        "intervals": {"ours": io, "attention": ia, "asnorm": iz},
    }
    op = REPORTS / "table1_seedvariance.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {op}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
