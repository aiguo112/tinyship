#!/usr/bin/env python
"""
TASK C0 — QiandaoEar22 cross-condition feasibility audit (NO training).

Gate: individual vessels with ≥20 clips in one condition (enroll) AND ≥20 in a
DIFFERENT disjoint condition (probe), for distance N/M/F and audibility S/M/W.

  python scripts/reproduce/audit_qiandao_crosscond.py
"""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MANIFEST = ROOT / "data" / "metadata" / "qiandaoear22_manifest.csv"

# Type / group / non-individual labels (exclude from individual gate)
EXCLUDE = {
    "SpeedBoat", "UUV", "KaiYuan", "Cargo", "FishBoat", "WorkShip", "MotorBoat",
    "Unknown", "ArtificialSignals", "Yacht", "Passenger", "Tanker", "Tug",
    "N_S", "M_S", "QianDao_M_MS", "No7_Slow_Down_1", "Background", "Noise",
}

# Filename: {ts}_{S|M}_{ShipName}_{N|M|F}_{S|M|W}...
# Name is alnum only (rejects multi-target & compounds).
PAT = re.compile(
    r"^(?P<ts>\d{14})_(?P<tm>[SM])_(?P<name>[A-Za-z][A-Za-z0-9]*)_(?P<dist>[NMF])_(?P<aud>[SMW])(?:_|$)"
)


def parse_name(fname: str):
    stem = Path(fname).stem
    # strip copy suffixes / label tails
    stem = re.sub(r"\s*-.*$", "", stem)
    stem = re.sub(r"_label__.*$", "", stem)
    m = PAT.match(stem)
    if not m:
        return None
    # single-target only — multi-target mixes identities
    if m.group("tm") != "S":
        return None
    return {
        "name": m.group("name"),
        "dist": m.group("dist"),
        "aud": m.group("aud"),
        "tm": m.group("tm"),
        "ts": m.group("ts"),
        "date": m.group("ts")[:8],
    }


def qualifies(counts: dict, min_n: int = 20) -> list[tuple[str, str, int, int]]:
    """Return (cond_a, cond_b, na, nb) pairs where both >= min_n and a!=b."""
    keys = [k for k, n in counts.items() if n >= min_n]
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pairs.append((a, b, counts[a], counts[b]))
    return pairs


def main():
    if not MANIFEST.exists():
        sys.exit(f"missing {MANIFEST}")
    df = pd.read_csv(MANIFEST)
    # unique content by sha256
    if "sha256" in df.columns:
        df = df.drop_duplicates(subset=["sha256"], keep="first")
    df = df[df["is_background"] != True].copy()  # noqa: E712

    rows = []
    parse_fail = 0
    for _, r in df.iterrows():
        p = parse_name(str(r["original_filename"]))
        if p is None:
            parse_fail += 1
            continue
        rows.append(p)
    meta = pd.DataFrame(rows)
    all_names = sorted(meta["name"].unique())
    individuals = [n for n in all_names if n not in EXCLUDE]
    excluded_present = [n for n in all_names if n in EXCLUDE]

    dist_gate = []
    aud_gate = []
    per_vessel = {}

    for name in individuals:
        sub = meta[meta["name"] == name]
        dc = sub["dist"].value_counts().to_dict()
        ac = sub["aud"].value_counts().to_dict()
        d_pairs = qualifies(dc, 20)
        a_pairs = qualifies(ac, 20)
        per_vessel[name] = {
            "n_clips": int(len(sub)),
            "dist_counts": {k: int(v) for k, v in dc.items()},
            "aud_counts": {k: int(v) for k, v in ac.items()},
            "dist_qualifying_pairs": [
                {"enroll": a, "probe": b, "n_enroll": na, "n_probe": nb}
                for a, b, na, nb in d_pairs
            ],
            "aud_qualifying_pairs": [
                {"enroll": a, "probe": b, "n_enroll": na, "n_probe": nb}
                for a, b, na, nb in a_pairs
            ],
        }
        if d_pairs:
            dist_gate.append(name)
        if a_pairs:
            aud_gate.append(name)

    n_dist = len(dist_gate)
    n_aud = len(aud_gate)
    # plan: >=~6 viable; <~4 limitation
    either = sorted(set(dist_gate) | set(aud_gate))
    verdict = "VIABLE" if len(either) >= 6 else ("MARGINAL" if len(either) >= 4 else "NOT_VIABLE")

    md = []
    md.append("# QiandaoEar22 cross-condition feasibility audit (C0)\n")
    md.append("**No training.** Gate: ≥20 clips in condition A AND ≥20 in disjoint condition B.\n")
    md.append(f"- Unique non-background clips (by sha256): **{len(meta)}**")
    md.append(f"- Parse failures: **{parse_fail}**")
    md.append(f"- All parsed name labels: **{len(all_names)}**")
    md.append(f"- Excluded type/group labels present: `{', '.join(excluded_present)}`")
    md.append(f"- Candidate **individual** names: **{len(individuals)}** → `{', '.join(individuals)}`\n")
    md.append("## Gate results\n")
    md.append(f"| Axis | Vessels with ≥20+≥20 in two conditions | Names |")
    md.append(f"|------|---------------------------------------:|-------|")
    md.append(f"| Distance (N/M/F) | **{n_dist}** | {', '.join(dist_gate) or '—'} |")
    md.append(f"| Audibility (S/M/W) | **{n_aud}** | {', '.join(aud_gate) or '—'} |")
    md.append(f"| Union | **{len(either)}** | {', '.join(either) or '—'} |\n")
    md.append(f"**Verdict: `{verdict}`** (plan: ≥~6 → C1 viable; <~4 → limitation only).\n")
    md.append("## Per-vessel counts (individuals)\n")
    for name in individuals:
        v = per_vessel[name]
        md.append(f"### `{name}` — {v['n_clips']} clips")
        md.append(f"- distance: `{v['dist_counts']}`")
        md.append(f"- audibility: `{v['aud_counts']}`")
        if v["dist_qualifying_pairs"]:
            md.append(f"- dist pairs: `{v['dist_qualifying_pairs']}`")
        if v["aud_qualifying_pairs"]:
            md.append(f"- aud pairs: `{v['aud_qualifying_pairs']}`")
        md.append("")

    out_md = REPORTS / "qiandao_crosscond_audit.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json = {
        "n_individual_names": len(individuals),
        "individuals": individuals,
        "excluded_present": excluded_present,
        "gate_distance_vessels": dist_gate,
        "gate_audibility_vessels": aud_gate,
        "gate_union_n": len(either),
        "gate_union": either,
        "verdict": verdict,
        "per_vessel": per_vessel,
    }
    (REPORTS / "qiandao_crosscond_audit.json").write_text(
        json.dumps(out_json, indent=2), encoding="utf-8"
    )
    print(f"GATE union vessels = {len(either)} → {verdict}")
    print(f"  distance: {n_dist} {dist_gate}")
    print(f"  audibility: {n_aud} {aud_gate}")
    print(f"wrote {out_md}")
    print("\nCOMMAND:")
    print(r"  python scripts\reproduce\audit_qiandao_crosscond.py")
    return verdict, len(either)


if __name__ == "__main__":
    main()
