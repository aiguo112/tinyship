"""Log-mel spectrogram features."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from src.features.config import load_feature_config


def compute_mel(
    y: np.ndarray,
    sr: int | None = None,
    *,
    n_mels: int | None = None,
    hop_length: int | None = None,
    n_fft: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = config or load_feature_config()
    sr = int(sr or cfg["sample_rate"])
    n_mels = int(n_mels or cfg.get("n_mels", cfg["n_bins"]))
    hop_length = int(hop_length or cfg["hop_length"])
    n_fft = int(n_fft or cfg["n_fft"])
    fmin = float(fmin if fmin is not None else cfg["fmin"])
    fmax = float(fmax if fmax is not None else cfg["fmax"])

    S = librosa.feature.melspectrogram(
        y=y.astype(np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        window=cfg.get("window", "hann"),
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)
