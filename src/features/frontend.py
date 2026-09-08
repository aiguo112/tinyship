"""
ShipNN feature front-end (CQT + MFCC), per ShipNN §4.1.

Config values are taken verbatim from the paper where stated, and flagged
where the paper is internally inconsistent. Nothing here is invented silently.

Paper-stated:
  sr = 32000, 1-second clips, Hann window, 125-sample hop
  CQT: 95 frequency bins spanning 18-4186 Hz  (=> ~12 bins/octave from fmin=18)
  MFCC: 13 coefficients, padded with the per-file MEAN to match 95 rows
  Best config: CQT + MFCC (stacked -> 2 channels)

KNOWN AMBIGUITY (do not paper over):
  The paper states hop=125 AND "<12,000 points/sec". 95*126 = 11,970 (<12k),
  but hop=125 on 32000 samples yields ~256 frames -> 95*256 = 24,320 (>12k).
  These conflict. We default to hop=125 (as stated) and expose N_FRAMES so the
  95x126 footprint claim can be tested. Resolve empirically on first run.
"""
from __future__ import annotations
import numpy as np
import librosa

# ---- paper config ----------------------------------------------------------
SR = 32000
HOP = 125                 # paper §4.1 (see ambiguity note above)
N_BINS = 95               # CQT frequency bins
FMIN = 18.0               # Hz
BINS_PER_OCTAVE = 12      # 18 * 2^(94/12) ~= 4.3 kHz, ~ paper's 4186 Hz top
N_MFCC = 13
N_FRAMES = None           # None = natural length; set 126 to test footprint claim
# ---------------------------------------------------------------------------


def _fix_frames(x: np.ndarray, n: int | None) -> np.ndarray:
    """Center-crop or edge-pad the time axis to n frames (no-op if n is None)."""
    if n is None:
        return x
    t = x.shape[-1]
    if t == n:
        return x
    if t > n:
        s = (t - n) // 2
        return x[..., s:s + n]
    pad = n - t
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad // 2, pad - pad // 2)], mode="edge")


def cqt_feature(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Log-magnitude CQT, shape (N_BINS, T)."""
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    C = librosa.cqt(y, sr=SR, hop_length=HOP, fmin=FMIN,
                    n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
    return librosa.amplitude_to_db(np.abs(C), ref=np.max).astype(np.float32)


def mfcc_feature(y: np.ndarray, sr: int = SR, rows: int = N_BINS) -> np.ndarray:
    """13 MFCCs padded with the file mean up to `rows`, shape (rows, T)."""
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    M = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP,
                             n_fft=min(2048, len(y))).astype(np.float32)
    if M.shape[0] < rows:                       # paper: pad with the mean value
        pad = np.full((rows - M.shape[0], M.shape[1]), float(M.mean()), np.float32)
        M = np.vstack([M, pad])
    return M


def extract(y: np.ndarray, sr: int = SR, feature: str = "cqt_mfcc") -> np.ndarray:
    """
    Returns a (C, F, T) float32 tensor.
      feature='cqt'       -> (1, 95, T)
      feature='mfcc'      -> (1, 95, T)
      feature='cqt_mfcc'  -> (2, 95, T)   <- ShipNN best config
    """
    if feature == "cqt":
        chans = [cqt_feature(y, sr)]
    elif feature == "mfcc":
        chans = [mfcc_feature(y, sr)]
    elif feature == "cqt_mfcc":
        c, m = cqt_feature(y, sr), mfcc_feature(y, sr)
        t = min(c.shape[-1], m.shape[-1])
        chans = [c[..., :t], m[..., :t]]
    else:
        raise ValueError(f"unknown feature '{feature}'")
    x = np.stack([_fix_frames(c, N_FRAMES) for c in chans], axis=0)
    # per-channel standardization (stabilizes training across the 100 classes)
    mu = x.mean(axis=(1, 2), keepdims=True)
    sd = x.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def in_channels(feature: str) -> int:
    return 2 if feature == "cqt_mfcc" else 1
