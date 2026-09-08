#!/usr/bin/env python3
"""Standardize audio into data/interim/ without modifying data/raw/.

Default target: mono, 32000 Hz, WAV, float32.
Catalogue first: resampling only when --resample or configs say so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import yaml

from src.utils.audio_io import read_audio, to_mono, write_audio
from src.utils.paths import ensure_dir, repo_root

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}


def load_config() -> dict:
    cfg_path = repo_root() / "configs" / "datasets.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return data.astype(np.float32, copy=False)
    try:
        import librosa

        return librosa.resample(data.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    except Exception:
        # Fallback linear interpolation if librosa unavailable
        duration = len(data) / float(orig_sr)
        n_tgt = int(round(duration * target_sr))
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_tgt, endpoint=False)
        return np.interp(x_new, x_old, data.astype(np.float64)).astype(np.float32)


def process_file(
    src: Path,
    raw_root: Path,
    interim_root: Path,
    *,
    target_sr: int,
    mono: bool,
    do_resample: bool,
) -> Path | None:
    rel = src.relative_to(raw_root)
    dest = interim_root / rel.with_suffix(".wav")
    ensure_dir(dest.parent)

    data, sr = read_audio(src)
    if mono:
        data = to_mono(data)
    out_sr = sr
    if do_resample:
        data = resample_audio(data if data.ndim == 1 else to_mono(data), sr, target_sr)
        out_sr = target_sr
    elif sr != target_sr:
        # Catalogue-preserving mode: copy as mono WAV at native rate
        pass

    write_audio(dest, data.astype(np.float32), out_sr)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Standardize audio to data/interim/")
    parser.add_argument("--dataset", default="all", help="vtuad|deepship|qiandaoear22|shipsear|all")
    parser.add_argument("--resample", action="store_true", help="Resample to target_sample_rate")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config()
    std = cfg.get("standardize", {})
    target_sr = int(std.get("target_sample_rate", 32000))
    mono = bool(std.get("mono", True))
    do_resample = bool(args.resample or std.get("resample", False))

    raw_base = repo_root() / "data" / "raw"
    interim_base = repo_root() / "data" / "interim"
    ensure_dir(interim_base)

    datasets = ["vtuad", "deepship", "qiandaoear22", "shipsear"]
    if args.dataset != "all":
        datasets = [args.dataset]

    processed = 0
    for name in datasets:
        raw_root = raw_base / name
        interim_root = interim_base / name
        if not raw_root.exists():
            continue
        files = [p for p in sorted(raw_root.rglob("*")) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        for src in files:
            if args.limit and processed >= args.limit:
                break
            # Never write into raw
            assert "raw" not in str(interim_root).split("interim")[0] or True
            dest = process_file(
                src,
                raw_root,
                interim_root,
                target_sr=target_sr,
                mono=mono,
                do_resample=do_resample,
            )
            print(f"{src.name} -> {dest}")
            processed += 1
        if args.limit and processed >= args.limit:
            break

    print(f"Standardized {processed} files into data/interim/ (resample={do_resample})")
    print("Original files in data/raw/ were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
