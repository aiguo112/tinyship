"""Tests for VTUAD metadata loading."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.vtuad_metadata import load_scenario_split_metadata, load_vtuad_metadata


def test_load_one_split():
    raw = ROOT / "data" / "raw" / "vtuad" / "inclusion_2000_exclusion_4000"
    if not (raw / "train" / "metadata.csv").exists():
        return
    df = load_scenario_split_metadata(raw, "train")
    assert not df.empty
    assert "MMSI" in df.columns
    assert "file_path" in df.columns
    assert df["duration_sec"].iloc[0] == 1.0


def test_load_all_metadata():
    raw = ROOT / "data" / "raw" / "vtuad"
    if not raw.exists():
        return
    df = load_vtuad_metadata(raw)
    if df.empty:
        return
    assert df["scenario"].nunique() >= 1
    assert set(df["split"].unique()) <= {"train", "validation", "test"}
