"""Basic smoke tests (no training, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.manifest import MANIFEST_COLUMNS, empty_manifest, rows_to_dataframe
from src.utils.audio_io import audio_stats, to_mono


def test_manifest_schema():
    df = empty_manifest()
    assert list(df.columns) == MANIFEST_COLUMNS
    df2 = rows_to_dataframe([{"dataset": "VTUAD", "file_path": "x.wav"}])
    assert df2.loc[0, "dataset"] == "VTUAD"
    assert df2.loc[0, "sha256"] is None


def test_audio_stats_near_zero():
    x = np.zeros(1000, dtype=np.float32)
    s = audio_stats(x)
    assert s["near_zero"] is True
    assert s["nan_count"] == 0


def test_to_mono():
    stereo = np.random.randn(100, 2).astype(np.float32)
    m = to_mono(stereo)
    assert m.ndim == 1 and len(m) == 100
