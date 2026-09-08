#!/usr/bin/env python
"""
TASK A — VTUAD method comparison under the LOCKED certified protocol.

Loads saved embeddings only (no training). Emits:
  reports/comparison_vtuad.json
  reports/comparison_vtuad.tex

  python scripts/reproduce/build_comparison_table.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.certified_verify import (  # noqa: E402
    K_ENROLL, PROBE_W, SEED,
    asnorm_set_eval, attention_set_eval, certified_mean_eval, clip_cosine_eval,
)

REPORTS = ROOT / "reports"


def load_emb(name: str):
    p = REPORTS / name
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return d["E"], np.asarray(d["ids"]).astype(str)


def main():
    rows = []

    rows.append({
        "method": "Closed-set (ShipNN-style softmax)",
        "eer": None,
        "top1": None,
        "note": "undefined on unseen vessels",
        "status": "undefined",
    })

    arc = load_emb("fingerprint_arcface_embeddings.npz")
    attn_emb = load_emb("fingerprint_attention_embeddings.npz")
    trip = load_emb("fingerprint_triplet_embeddings.npz")

    if arc is None:
        sys.exit("missing fingerprint_arcface_embeddings.npz")

    E_a, ids_a = arc
    print("DISJOINTNESS: scoring ArcFace embeddings under certified protocols...")
    r_clip = clip_cosine_eval(E_a, ids_a, seed=SEED)
    print(f"  clip_cosine PASS ships={r_clip['n_ships']} EER={r_clip['eer']} top1={r_clip['top1']}")
    rows.append({
        "method": "ArcFace + clip cosine",
        "eer": r_clip["eer"], "top1": r_clip["top1"],
        "note": "clip-level backend (inherent)", "status": "ok",
        "detail": r_clip,
    })

    r_proto = certified_mean_eval(E_a, ids_a, k_enroll=K_ENROLL, probe_w=1, seed=SEED)
    print(f"  prototype_single PASS EER={r_proto['eer']} top1={r_proto['top1']}")
    rows.append({
        "method": "ArcFace + prototype enroll ($k{=}20$, single-clip probe)",
        "eer": r_proto["eer"], "top1": r_proto["top1"],
        "note": "backend: mean proto; probe_w=1", "status": "ok",
        "detail": r_proto,
    })

    r_asn = asnorm_set_eval(E_a, ids_a, k_enroll=K_ENROLL, probe_w=PROBE_W, seed=SEED)
    print(f"  asnorm PASS EER={r_asn['eer']} top1={r_asn['top1']}")
    rows.append({
        "method": "ArcFace + prototype + AS-norm ($k{=}20$, $w{=}3$)",
        "eer": r_asn["eer"], "top1": r_asn["top1"],
        "note": "backend: AS-norm only change", "status": "ok",
        "detail": r_asn,
    })

    r_pool = certified_mean_eval(E_a, ids_a, k_enroll=K_ENROLL, probe_w=PROBE_W, seed=SEED)
    print(f"  temporal_mean PASS EER={r_pool['eer']} top1={r_pool['top1']}")
    rows.append({
        "method": "ArcFace + temporal mean pooling ($k{=}20$, $w{=}3$)",
        "eer": r_pool["eer"], "top1": r_pool["top1"],
        "note": "certified mean protocol on ArcFace emb", "status": "ok",
        "detail": r_pool,
    })

    if attn_emb is not None:
        E_t, ids_t = attn_emb
        ckpt_p = REPORTS / "fingerprint_attention_ckpt.pt"
        if ckpt_p.exists():
            import torch
            from src.models.fingerprint import GatedAttentionPool
            ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            attn = GatedAttentionPool(ck.get("config", {}).get("emb_dim", 128), hidden=64)
            attn.load_state_dict(ck["attention"])
            r_att = attention_set_eval(E_t, ids_t, attn, "cpu", K_ENROLL, PROBE_W, SEED)
            print(f"  attention PASS EER={r_att['eer']} top1={r_att['top1']}")
            rows.append({
                "method": "Sub-center ArcFace + attention set pooling",
                "eer": r_att["eer"], "top1": r_att["top1"],
                "note": "attention pool; same split", "status": "ok",
                "detail": r_att,
            })
        else:
            rows.append({
                "method": "Sub-center ArcFace + attention set pooling",
                "eer": None, "top1": None, "note": "missing ckpt", "status": "TODO",
            })

        r_ours = certified_mean_eval(E_t, ids_t, K_ENROLL, PROBE_W, SEED)
        print(f"  ours_mean PASS EER={r_ours['eer']} top1={r_ours['top1']}")
        rows.append({
            "method": r"\textbf{Sub-center ArcFace + mean set pooling (ours)}",
            "eer": r_ours["eer"], "top1": r_ours["top1"],
            "note": "certified; matches certify_attention_eer", "status": "ok",
            "detail": r_ours, "ours": True,
        })
    else:
        rows.append({
            "method": "Sub-center ArcFace + attention set pooling",
            "eer": None, "top1": None, "note": "missing embeddings", "status": "TODO",
        })
        rows.append({
            "method": r"\textbf{Sub-center ArcFace + mean set pooling (ours)}",
            "eer": None, "top1": None, "note": "missing embeddings", "status": "TODO",
        })

    if trip is not None:
        E_tr, ids_tr = trip
        r_tr = certified_mean_eval(E_tr, ids_tr, K_ENROLL, PROBE_W, SEED)
        print(f"  triplet PASS EER={r_tr['eer']} top1={r_tr['top1']}")
        # insert before ours if possible
        rows.insert(-1 if rows[-1].get("ours") else len(rows), {
            "method": "Triplet + temporal mean pooling ($k{=}20$, $w{=}3$)",
            "eer": r_tr["eer"], "top1": r_tr["top1"],
            "note": "same encoder/protocol", "status": "ok",
            "detail": r_tr,
        })
    else:
        rows.insert(-1 if (rows and rows[-1].get("ours")) else len(rows), {
            "method": "Triplet + temporal mean pooling ($k{=}20$, $w{=}3$)",
            "eer": None, "top1": None, "note": "embeddings not trained yet", "status": "TODO",
        })

    out = {
        "protocol": {
            "k_enroll": K_ENROLL, "probe_w": PROBE_W, "seed": SEED,
            "note": "All set-level rows use certified enroll/probe split; "
                    "clip-cosine and probe_w=1 / AS-norm are labeled backend variants.",
        },
        "rows": rows,
    }
    jp = REPORTS / "comparison_vtuad.json"
    jp.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    # LaTeX table body
    lines = [
        r"% Auto-generated by scripts/reproduce/build_comparison_table.py — do not hand-edit",
        r"\begin{table}[t]",
        r"\caption{Open-set method comparison on the $22$ MMSI-disjoint unseen VTUAD vessels. "
        r"Unless noted, enrollment $k{=}20$ and probe window $w{=}3$ under the certified "
        r"clip-disjoint protocol (seed~$0$). Closed-set softmax is undefined on unseen identities.}",
        r"\label{tab:comparison}",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & EER & top-$1$ \\",
        r"\midrule",
    ]
    for r in rows:
        name = r["method"]
        if r["status"] == "undefined":
            eer_s, t1_s = r"\emph{undefined}", "---"
            lines.append(f"{name} & {eer_s} & ${t1_s}$ \\\\")
            continue
        elif r["status"] == "TODO":
            eer_s, t1_s = "TODO", "TODO"
        else:
            eer_s = f"{r['eer']:.3f}" if r["eer"] is not None else "---"
            t1_s = f"{r['top1']:.3f}" if r["top1"] is not None else "---"
            if r.get("ours"):
                eer_s = rf"\textbf{{{eer_s}}}"
                t1_s = rf"\textbf{{{t1_s}}}"
        lines.append(f"{name} & ${eer_s}$ & ${t1_s}$ \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    tp = REPORTS / "comparison_vtuad.tex"
    tp.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== COMPARISON TABLE ===")
    for r in rows:
        print(f"  {r['status']:9} EER={str(r.get('eer')):8} top1={str(r.get('top1')):8} | {r['method'][:60]}")
    print(f"\nwrote {jp}")
    print(f"wrote {tp}")
    print("\nCOMMAND:")
    print(r"  . .\scripts\use_tinyship.ps1")
    print(r"  python scripts\reproduce\build_comparison_table.py")


if __name__ == "__main__":
    main()
