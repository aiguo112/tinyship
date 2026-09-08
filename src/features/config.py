"""Feature config helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import repo_root


def load_feature_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else repo_root() / "configs" / "datasets.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    feats = dict(cfg.get("features", {}))
    # ShipNN defaults
    feats.setdefault("sample_rate", 32000)
    feats.setdefault("hop_length", 125)
    feats.setdefault("window", "hann")
    feats.setdefault("fmin", 18.0)
    feats.setdefault("fmax", 4186.0)
    feats.setdefault("n_bins", 95)
    feats.setdefault("n_mfcc", 13)
    feats.setdefault("n_mels", 95)
    feats.setdefault("n_fft", 2048)
    return feats
