"""Gammatone filterbank features (practical approximation).

Uses a ERB-spaced gammatone-like mel alternative via librosa filters when
`spafe` is unavailable; prefers spafe.fbanks.gammatone_fbanks if installed.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np
from numpy.typing import NDArray

from src.features.config import load_feature_config


def _erb_space(fmin: float, fmax: float, n: int) -> NDArray[np.float64]:
    # Glasberg & Moore ERB
    def hz_to_erb(f: float) -> float:
        return 21.4 * np.log10(0.00437 * f + 1.0)

    def erb_to_hz(e: float) -> float:
        return (10 ** (e / 21.4) - 1.0) / 0.00437

    e_min, e_max = hz_to_erb(fmin), hz_to_erb(fmax)
    return np.array([erb_to_hz(e) for e in np.linspace(e_min, e_max, n)], dtype=np.float64)


def _gammatone_filterbank(
    sr: int,
    n_fft: int,
    n_bins: int,
    fmin: float,
    fmax: float,
) -> NDArray[np.float32]:
    try:
        from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks

        fbanks, _ = gammatone_filter_banks(
            nfilts=n_bins,
            nfft=n_fft,
            fs=sr,
            low_freq=fmin,
            high_freq=fmax,
        )
        return np.asarray(fbanks, dtype=np.float32)
    except Exception:
        # Fallback: triangular filters centered on ERB-spaced frequencies
        freqs = _erb_space(fmin, fmax, n_bins)
        fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        fb = np.zeros((n_bins, len(fft_freqs)), dtype=np.float32)
        for i, fc in enumerate(freqs):
            # simple Gaussian-shaped approximation around center
            bw = max(fc * 0.1, 5.0)
            fb[i] = np.exp(-0.5 * ((fft_freqs - fc) / bw) ** 2)
        fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-8)
        return fb


def compute_gammatone(
    y: np.ndarray,
    sr: int | None = None,
    *,
    n_bins: int | None = None,
    hop_length: int | None = None,
    n_fft: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = config or load_feature_config()
    sr = int(sr or cfg["sample_rate"])
    n_bins = int(n_bins or cfg["n_bins"])
    hop_length = int(hop_length or cfg["hop_length"])
    n_fft = int(n_fft or cfg["n_fft"])
    fmin = float(fmin if fmin is not None else cfg["fmin"])
    fmax = float(fmax if fmax is not None else min(cfg["fmax"], sr / 2 - 1))

    S = np.abs(
        librosa.stft(
            y.astype(np.float32),
            n_fft=n_fft,
            hop_length=hop_length,
            window=cfg.get("window", "hann"),
        )
    )
    fb = _gammatone_filterbank(sr, n_fft, n_bins, fmin, fmax)
    # fb shape (n_bins, n_fft//2+1)
    gt = fb @ S
    return librosa.power_to_db(gt**2 + 1e-10, ref=np.max).astype(np.float32)
