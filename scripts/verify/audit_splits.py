#!/usr/bin/env python3
"""Audit leakage across split protocols. No training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.data.splits import load_splits_config, prepare_catalog
from src.utils.paths import repo_root

PROTOCOLS = ("random", "mmsi", "session", "temporal", "scenario", "official")


def load_protocol(out_dir: Path, protocol: str) -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        path = out_dir / f"{protocol}_{split}.csv"
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def nonzero_mmsi(s: pd.Series) -> set[int]:
    vals = s.dropna().astype(int)
    return set(int(x) for x in vals if x != 0)


def set_overlap(a: set, b: set) -> int:
    return len(a & b)


def audit_protocol(df: pd.DataFrame, protocol: str) -> dict:
    if df.empty:
        return {"protocol": protocol, "error": "missing splits"}

    trains = df[df["split"] == "train"]
    vals = df[df["split"] == "val"]
    tests = df[df["split"] == "test"]

    m_tr, m_va, m_te = map(nonzero_mmsi, [trains["MMSI"], vals["MMSI"], tests["MMSI"]])
    s_tr = set(trains["sub_init"].dropna().astype(int))
    s_va = set(vals["sub_init"].dropna().astype(int))
    s_te = set(tests["sub_init"].dropna().astype(int))
    f_tr = set(trains["file_path"].astype(str))
    f_va = set(vals["file_path"].astype(str))
    f_te = set(tests["file_path"].astype(str))

    # SHA256 overlaps (non-null)
    def hashes(part: pd.DataFrame) -> set[str]:
        if "sha256" not in part.columns:
            return set()
        return set(part["sha256"].dropna().astype(str))

    h_tr, h_va, h_te = map(hashes, [trains, vals, tests])

    d_tr = set(trains["date"].dropna().astype(int))
    d_va = set(vals["date"].dropna().astype(int))
    d_te = set(tests["date"].dropna().astype(int))

    sc_tr = set(trains["scenario"].dropna().astype(str))
    sc_va = set(vals["scenario"].dropna().astype(str))
    sc_te = set(tests["scenario"].dropna().astype(str))

    def ship_stats(part: pd.DataFrame) -> dict:
        ships = part[~part.get("is_background", False).astype(bool) & (part["MMSI"].fillna(0).astype(int) != 0)]
        if "is_background" not in part.columns:
            ships = part[part["MMSI"].fillna(0).astype(int) != 0]
        bg = part[(part["MMSI"].fillna(0).astype(int) == 0) | (part.get("label") == "background")]
        return {
            "clips": int(len(part)),
            "ships": int(ships["MMSI"].nunique()) if len(ships) else 0,
            "sessions": int(part["sub_init"].nunique()),
            "background_clips": int(len(bg)),
            "labels": part["label"].value_counts().to_dict() if "label" in part.columns else {},
        }

    return {
        "protocol": protocol,
        "train": ship_stats(trains),
        "val": ship_stats(vals),
        "test": ship_stats(tests),
        "mmsi_overlap": {
            "train_val": set_overlap(m_tr, m_va),
            "train_test": set_overlap(m_tr, m_te),
            "val_test": set_overlap(m_va, m_te),
        },
        "session_overlap": {
            "train_val": set_overlap(s_tr, s_va),
            "train_test": set_overlap(s_tr, s_te),
            "val_test": set_overlap(s_va, s_te),
        },
        "file_overlap": {
            "train_val": set_overlap(f_tr, f_va),
            "train_test": set_overlap(f_tr, f_te),
            "val_test": set_overlap(f_va, f_te),
        },
        "sha256_overlap": {
            "train_val": set_overlap(h_tr, h_va),
            "train_test": set_overlap(h_tr, h_te),
            "val_test": set_overlap(h_va, h_te),
        },
        "date_overlap": {
            "train_val": set_overlap(d_tr, d_va),
            "train_test": set_overlap(d_tr, d_te),
            "val_test": set_overlap(d_va, d_te),
            "train_date_range": [int(min(d_tr)), int(max(d_tr))] if d_tr else None,
            "val_date_range": [int(min(d_va)), int(max(d_va))] if d_va else None,
            "test_date_range": [int(min(d_te)), int(max(d_te))] if d_te else None,
        },
        "scenario_overlap": {
            "train_val": sorted(sc_tr & sc_va),
            "train_test": sorted(sc_tr & sc_te),
            "val_test": sorted(sc_va & sc_te),
            "train_scenarios": sorted(sc_tr),
            "val_scenarios": sorted(sc_va),
            "test_scenarios": sorted(sc_te),
        },
        "mmsi_leakage": any(
            [
                set_overlap(m_tr, m_va),
                set_overlap(m_tr, m_te),
                set_overlap(m_va, m_te),
            ]
        ),
        "session_leakage": any(
            [
                set_overlap(s_tr, s_va),
                set_overlap(s_tr, s_te),
                set_overlap(s_va, s_te),
            ]
        ),
    }


def overlap_matrix(df: pd.DataFrame, column: str, nonzero_only: bool = False) -> pd.DataFrame:
    """Pairwise |A ∩ B| for train/val/test group membership."""
    splits = ["train", "val", "test"]
    sets = {}
    for sp in splits:
        part = df[df["split"] == sp][column].dropna()
        if nonzero_only:
            part = part.astype(int)
            part = part[part != 0]
        sets[sp] = set(part.astype(str) if not nonzero_only else part.astype(int))
    mat = pd.DataFrame(index=splits, columns=splits, dtype=int)
    for a in splits:
        for b in splits:
            mat.loc[a, b] = len(sets[a] & sets[b])
    return mat


def protocol_statistics(df: pd.DataFrame, protocol: str) -> list[dict]:
    rows = []
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        ships = part[part["MMSI"].fillna(0).astype(int) != 0]
        bg = part[part["MMSI"].fillna(0).astype(int) == 0]
        per_mmsi = ships.groupby("MMSI").size() if len(ships) else pd.Series(dtype=float)
        per_sess = part.groupby("sub_init").size() if len(part) else pd.Series(dtype=float)
        label_counts = part["label"].value_counts().to_dict() if "label" in part.columns else {}
        rows.append(
            {
                "protocol": protocol,
                "split": split,
                "n_clips": len(part),
                "n_ships_mmsi": int(ships["MMSI"].nunique()) if len(ships) else 0,
                "n_sessions": int(part["sub_init"].nunique()) if len(part) else 0,
                "n_background": int(len(bg)),
                "samples_per_mmsi_mean": float(per_mmsi.mean()) if len(per_mmsi) else 0.0,
                "samples_per_mmsi_median": float(per_mmsi.median()) if len(per_mmsi) else 0.0,
                "samples_per_session_mean": float(per_sess.mean()) if len(per_sess) else 0.0,
                **{f"label_{k}": int(v) for k, v in label_counts.items()},
            }
        )
    return rows


def write_report_md(audits: list[dict], catalog: pd.DataFrame, path: Path, temporal_boundaries: dict | None) -> None:
    ships = catalog[catalog["MMSI"].fillna(0).astype(int) != 0]
    bg = catalog[catalog["MMSI"].fillna(0).astype(int) == 0]
    lines = [
        "# VTUAD Split Audit Report",
        "",
        "Generated by `scripts/verify/audit_splits.py`. **No models trained.**",
        "",
        "## Dataset statistics",
        "",
        f"- total recordings (metadata rows): **{len(catalog)}**",
        f"- total ships (nonzero MMSI): **{ships['MMSI'].nunique()}**",
        f"- total sessions (`sub_init`): **{catalog['sub_init'].nunique()}**",
        f"- background / MMSI=0 recordings: **{len(bg)}**",
        f"- date range: **{int(catalog['date'].min())}** – **{int(catalog['date'].max())}** ({catalog['date'].nunique()} unique dates)",
        "",
        "### Scenario distribution",
        "",
        "| Scenario | Clips | Inclusion / exclusion (m) | Distance interpretation |",
        "|---|---:|---|---|",
        "| inclusion_2000_exclusion_4000 | "
        f"{int((catalog.scenario=='inclusion_2000_exclusion_4000').sum())} | 2000 / 4000 | ~2–4 km band |",
        "| inclusion_3000_exclusion_5000 | "
        f"{int((catalog.scenario=='inclusion_3000_exclusion_5000').sum())} | 3000 / 5000 | ~3–5 km band |",
        "| inclusion_4000_exclusion_6000 | "
        f"{int((catalog.scenario=='inclusion_4000_exclusion_6000').sum())} | 4000 / 6000 | ~4–6 km band |",
        "",
    ]
    if temporal_boundaries:
        lines += [
            "### Temporal date boundaries",
            "",
            "```json",
            json.dumps(temporal_boundaries, indent=2),
            "```",
            "",
        ]

    lines += [
        "## Protocol table",
        "",
        "| Protocol | Train ships | Val ships | Test ships | MMSI leakage | Session leakage |",
        "|---|---:|---:|---:|---|---|",
    ]
    for a in audits:
        if a.get("error"):
            lines.append(f"| {a['protocol']} |  |  |  | error | error |")
            continue
        lines.append(
            f"| {a['protocol']} | {a['train']['ships']} | {a['val']['ships']} | {a['test']['ships']} | "
            f"{'YES' if a['mmsi_leakage'] else 'NO'} | {'YES' if a['session_leakage'] else 'NO'} |"
        )

    lines += [
        "",
        "## Scientific notes",
        "",
        "- **Random / Official**: clip-level leakage expected; **not** valid for ship-independent claims.",
        "- **MMSI**: ship-disjoint for nonzero MMSI; suitable for **ship-independent** evaluation "
        "(not closed-set ID of the same ships).",
        "- **Session**: recording-session disjoint; may still leak ships if a ship has multiple sessions.",
        "- **Temporal**: chronological; may leak ships that reappear across dates.",
        "- **Scenario**: train near / val mid / test far; SHA256-deduped so identical audio is not in multiple splits; "
        "ships may still overlap across distance bands.",
        "",
        "### Fingerprinting reminder",
        "",
        "- Closed-set classification ≠ unseen-ship identification.",
        "- MMSI-disjoint splits evaluate ship-independent generalization, not conventional closed-set ID.",
        "- Future verification/retrieval will need enrollment + query partitions (not implemented here).",
        "",
        "## Recommendations",
        "",
        "- **Primary protocol:** `mmsi` (ship-independent classification / domain for later fingerprinting setup).",
        "- **Secondary protocol:** `session` (stricter against neighboring-clip / session leakage).",
        "- Keep `random` only as a **ShipNN-style baseline** for leakage comparison.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cfg = load_splits_config()
    out_dir = repo_root() / cfg.get("output_dir", "data/processed/splits")
    reports = repo_root() / cfg.get("reports_dir", "reports")
    reports.mkdir(parents=True, exist_ok=True)

    catalog = prepare_catalog(
        quality_csv=repo_root() / cfg.get("quality_report", "data/metadata/data_quality_report.csv")
    )
    audits = []
    stats_rows = []
    # Prefer mmsi protocol for matrices; also write for random
    for protocol in PROTOCOLS:
        df = load_protocol(out_dir, protocol)
        if df.empty:
            print(f"SKIP missing {protocol}")
            continue
        a = audit_protocol(df, protocol)
        audits.append(a)
        stats_rows.extend(protocol_statistics(df, protocol))
        print(
            f"{protocol}: mmsi_leak={a['mmsi_leakage']} session_leak={a['session_leakage']} "
            f"ships={a['train']['ships']}/{a['val']['ships']}/{a['test']['ships']}"
        )

    # Overlap matrices for primary protocols
    for protocol, col, nonzero, name in [
        ("random", "MMSI", True, "mmsi_overlap_matrix_random.csv"),
        ("mmsi", "MMSI", True, "mmsi_overlap_matrix.csv"),
        ("random", "sub_init", False, "session_overlap_matrix_random.csv"),
        ("session", "sub_init", False, "session_overlap_matrix.csv"),
    ]:
        df = load_protocol(out_dir, protocol)
        if df.empty:
            continue
        mat = overlap_matrix(df, col, nonzero_only=nonzero)
        mat.to_csv(reports / name)
        print(f"Wrote {reports / name}")

    pd.DataFrame(stats_rows).to_csv(reports / "split_statistics.csv", index=False)
    with open(reports / "split_audit.json", "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2)

    temporal_boundaries = (
        cfg.get("protocols", {}).get("temporal", {}).get("date_boundaries")
        if isinstance(cfg.get("protocols"), dict)
        else None
    )
    write_report_md(audits, catalog, reports / "split_audit.md", temporal_boundaries)
    print(f"Wrote {reports / 'split_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
