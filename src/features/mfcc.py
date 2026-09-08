"""MFCC features (ShipNN-compatible: 13 coefficients by default)."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from src.features.config import load_feature_config


def compute_mfcc(
    y: np.ndarray,
    sr: int | None = None,
    *,
    n_mfcc: int | None = None,
    hop_length: int | None = None,
    n_fft: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = config or load_feature_config()
    sr = int(sr or cfg["sample_rate"])
    n_mfcc = int(n_mfcc or cfg["n_mfcc"])
    hop_length = int(hop_length or cfg["hop_length"])
    n_fft = int(n_fft or cfg["n_fft"])
    fmin = float(fmin if fmin is not None else cfg["fmin"])
    fmax = float(fmax if fmax is not None else cfg["fmax"])

    return librosa.feature.mfcc(
        y=y.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
    ).astype(np.float32)
