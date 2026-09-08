"""Linear spectrogram features."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from src.features.config import load_feature_config


def compute_spectrogram(
    y: np.ndarray,
    sr: int | None = None,
    *,
    hop_length: int | None = None,
    n_fft: int | None = None,
    window: str | None = None,
    config: dict[str, Any] | None = None,
    to_db: bool = True,
) -> np.ndarray:
    cfg = config or load_feature_config()
    hop_length = int(hop_length or cfg["hop_length"])
    n_fft = int(n_fft or cfg["n_fft"])
    window = window or cfg.get("window", "hann")

    S = np.abs(
        librosa.stft(
            y.astype(np.float32),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
        )
    )
    if to_db:
        S = librosa.amplitude_to_db(S, ref=np.max)
    return S.astype(np.float32)
