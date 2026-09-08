#!/usr/bin/env python3
"""Verify audio integrity across data/raw (and optional manifests).

Checks: readable audio, sample rate, channels, duration, NaN/Inf,
clipping %, near-zero signals, corruption, duplicates (SHA256).

Writes: data/metadata/data_quality_report.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.utils.audio_io import audio_stats, read_audio
from src.utils.hashing import sha256_file
from src.utils.paths import ensure_dir, repo_root

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}


def discover_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS and p.name != ".gitkeep":
                files.append(p)
    return sorted(files)


def verify_one(path: Path) -> dict:
    rel = str(path.relative_to(repo_root())) if path.is_relative_to(repo_root()) else str(path)
    row = {
        "file_path": rel,
        "original_filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "sha256": None,
        "readable": False,
        "sample_rate": None,
        "channels": None,
        "duration_seconds": None,
        "nan_count": None,
        "inf_count": None,
        "clipping_pct": None,
        "near_zero": None,
        "peak": None,
        "rms": None,
        "corrupted": True,
        "error": None,
        "duplicate_of": None,
    }
    try:
        row["sha256"] = sha256_file(path)
        data, sr = read_audio(path)
        stats = audio_stats(data)
        channels = 1 if data.ndim == 1 else int(data.shape[-1])
        n = data.shape[0]
        row.update(
            {
                "readable": True,
                "sample_rate": sr,
                "channels": channels,
                "duration_seconds": float(n) / float(sr) if sr else None,
                "nan_count": stats["nan_count"],
                "inf_count": stats["inf_count"],
                "clipping_pct": stats["clipping_pct"],
                "near_zero": stats["near_zero"],
                "peak": stats["peak"],
                "rms": stats["rms"],
                "corrupted": bool(stats["nan_count"] or stats["inf_count"] or n == 0),
                "error": None,
            }
        )
    except Exception as e:  # noqa: BLE001
        row["error"] = str(e)
        row["corrupted"] = True
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify audio files")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["data/raw"],
        help="Roots to scan (relative to repo root)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max files to verify (0 = all)",
    )
    args = parser.parse_args()

    roots = [repo_root() / r for r in args.roots]
    files = discover_files(roots)
    if args.limit > 0:
        files = files[: args.limit]

    rows = []
    hash_to_first: dict[str, str] = {}
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        row = verify_one(path)
        digest = row.get("sha256")
        if digest:
            if digest in hash_to_first:
                row["duplicate_of"] = hash_to_first[digest]
            else:
                hash_to_first[digest] = row["file_path"]
        rows.append(row)

    out = repo_root() / "data" / "metadata" / "data_quality_report.csv"
    ensure_dir(out.parent)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    n_dup = int(df["duplicate_of"].notna().sum()) if len(df) else 0
    n_bad = int(df["corrupted"].sum()) if len(df) else 0
    print(f"\nWrote {out}")
    print(f"Files: {len(df)} | corrupted: {n_bad} | duplicates: {n_dup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
