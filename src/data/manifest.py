"""Unified dataset manifest schema and I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

MANIFEST_COLUMNS: list[str] = [
    "dataset",
    "file_path",
    "recording_id",
    "ship_id",
    "mmsi",
    "ship_type",
    "duration_seconds",
    "sample_rate",
    "channels",
    "source_location",
    "recording_date",
    "session_id",
    "distance",
    "audibility",
    "target_count",
    "is_background",
    "original_filename",
    "sha256",
    "file_size_bytes",
]


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Fill missing fields with None rather than inventing values."""
    out: dict[str, Any] = {}
    for col in MANIFEST_COLUMNS:
        val = row.get(col, None)
        if val == "" or (isinstance(val, float) and pd.isna(val)):
            val = None
        out[col] = val
    return out


def rows_to_dataframe(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    normalized = [normalize_row(r) for r in rows]
    if not normalized:
        return empty_manifest()
    return pd.DataFrame(normalized, columns=MANIFEST_COLUMNS)


def write_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = None
    out = df[MANIFEST_COLUMNS].copy()
    out.to_csv(path, index=False)
    return path


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return empty_manifest()
    df = pd.read_csv(path)
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[MANIFEST_COLUMNS]


def merge_manifests(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return empty_manifest()
    parts = []
    for df in dfs:
        if df is None or df.empty:
            continue
        for col in MANIFEST_COLUMNS:
            if col not in df.columns:
                df = df.copy()
                df[col] = None
        parts.append(df[MANIFEST_COLUMNS])
    if not parts:
        return empty_manifest()
    return pd.concat(parts, ignore_index=True)
