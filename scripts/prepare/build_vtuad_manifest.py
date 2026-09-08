#!/usr/bin/env python3
"""Build unified VTUAD manifest from official metadata.csv files."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.manifest import MANIFEST_COLUMNS, rows_to_dataframe, write_manifest
from src.data.vtuad_metadata import join_quality_report, load_vtuad_metadata
from src.utils.paths import repo_root


def main() -> int:
    raw = repo_root() / "data" / "raw" / "vtuad"
    meta = load_vtuad_metadata(raw)
    if meta.empty:
        print("No VTUAD metadata found.")
        return 1

    quality = repo_root() / "data" / "metadata" / "data_quality_report.csv"
    if quality.exists():
        meta = join_quality_report(meta, quality)

    rows = []
    for _, r in meta.iterrows():
        rows.append(
            {
                "dataset": "VTUAD",
                "file_path": r["file_path"],
                "recording_id": str(int(r["file_index"])),
                "ship_id": str(int(r["MMSI"])) if pd.notna(r["MMSI"]) else None,
                "mmsi": str(int(r["MMSI"])) if pd.notna(r["MMSI"]) else None,
                "ship_type": r["label"],
                "duration_seconds": r.get("duration_seconds") or r.get("duration_sec"),
                "sample_rate": r.get("sample_rate"),
                "channels": 1,
                "source_location": "Georgia Strait, ONC",
                "recording_date": str(int(r["date"])) if pd.notna(r["date"]) else None,
                "session_id": str(int(r["session_id"])) if pd.notna(r["session_id"]) else None,
                "distance": r["distance_band"],
                "audibility": None,
                "target_count": 1 if r["label"] != "background" else 0,
                "is_background": r["label"] == "background",
                "original_filename": f"{int(r['file_index'])}.wav",
                "sha256": r.get("sha256"),
                "file_size_bytes": None,
            }
        )

    out_path = repo_root() / "data" / "metadata" / "vtuad_manifest.csv"
    write_manifest(rows_to_dataframe(rows), out_path)
    print(f"Wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
