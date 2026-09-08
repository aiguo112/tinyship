"""Constant-Q transform features (ShipNN-compatible defaults)."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from src.features.config import load_feature_config


def compute_cqt(
    y: np.ndarray,
    sr: int | None = None,
    *,
    hop_length: int | None = None,
    fmin: float | None = None,
    n_bins: int | None = None,
    window: str | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = config or load_feature_config()
    sr = int(sr or cfg["sample_rate"])
    hop_length = int(hop_length or cfg["hop_length"])
    fmin = float(fmin if fmin is not None else cfg["fmin"])
    n_bins = int(n_bins or cfg["n_bins"])
    window = window or cfg.get("window", "hann")

    # Approximate ShipNN band with n_bins spanning fmin..fmax via bins_per_octave
    fmax = float(cfg.get("fmax", 4186.0))
    if fmin <= 0:
        fmin = 18.0
    # bins_per_octave such that fmin * 2^(n_bins/bpo) ~= fmax
    import math

    bpo = max(1, int(round(n_bins / max(math.log2(fmax / fmin), 1e-6))))
    C = librosa.cqt(
        y.astype(np.float32),
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bpo,
        window=window,
    )
    return np.abs(C).astype(np.float32)
