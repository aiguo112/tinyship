"""Common download helpers (resumable HTTP, status JSON)."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Allow running scripts without installing the package
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.paths import ensure_dir, repo_root  # noqa: E402


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    chunk_size: int = 1 << 20,
) -> dict[str, Any]:
    """Resumable download via HTTP Range. Returns status dict."""
    ensure_dir(dest.parent)
    existing = dest.stat().st_size if dest.exists() else 0
    headers: dict[str, str] = {}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, dest.open(mode) as out:
            # If server ignores Range and returns 200, restart
            if existing > 0 and getattr(resp, "status", 200) == 200:
                out.close()
                dest.unlink(missing_ok=True)
                return download_file(url, dest, expected_sha256=expected_sha256, chunk_size=chunk_size)
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.HTTPError as e:
        if e.code == 416 and dest.exists():
            pass  # already complete
        else:
            raise

    digest = sha256_file(dest)
    ok = expected_sha256 is None or digest.lower() == expected_sha256.lower()
    return {
        "url": url,
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "sha256": digest,
        "sha256_ok": ok,
        "complete": ok if expected_sha256 else True,
    }


def print_missing_credentials(missing: list[str], instructions: list[str]) -> None:
    print("Missing credentials / configuration:")
    for m in missing:
        print(f"  - {m}")
    print("\nManual steps:")
    for i, step in enumerate(instructions, 1):
        print(f"  {i}. {step}")


def project_data(*parts: str) -> Path:
    return repo_root().joinpath("data", *parts)
