"""Audio I/O helpers. Never write into data/raw/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio as float32 array shaped (samples,) or (samples, channels)."""
    data, sr = sf.read(str(path), always_2d=False, dtype="float32")
    return data, int(sr)


def write_audio(path: str | Path, data: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    if "data" in path.parts and "raw" in path.parts:
        # Hard guard: never overwrite raw corpus
        raw_idx = path.parts.index("raw")
        if raw_idx > 0 and path.parts[raw_idx - 1] == "data":
            raise RuntimeError(f"Refusing to write into data/raw/: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype="FLOAT")
    return path


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.float32, copy=False)
    return np.mean(data, axis=-1).astype(np.float32)


def audio_stats(data: np.ndarray) -> dict[str, Any]:
    x = np.asarray(data, dtype=np.float64)
    if x.size == 0:
        return {
            "n_samples": 0,
            "nan_count": 0,
            "inf_count": 0,
            "clipping_pct": 0.0,
            "near_zero": True,
            "peak": 0.0,
            "rms": 0.0,
        }
    nan_count = int(np.isnan(x).sum())
    inf_count = int(np.isinf(x).sum())
    finite = x[np.isfinite(x)]
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    rms = float(np.sqrt(np.mean(finite**2))) if finite.size else 0.0
    clip = float((np.abs(finite) >= 0.999).mean() * 100.0) if finite.size else 0.0
    near_zero = bool(peak < 1e-6 or rms < 1e-8)
    return {
        "n_samples": int(x.size),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "clipping_pct": clip,
        "near_zero": near_zero,
        "peak": peak,
        "rms": rms,
    }
