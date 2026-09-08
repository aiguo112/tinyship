"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return repository root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def resolve_under_root(*parts: str | Path) -> Path:
    return repo_root().joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
