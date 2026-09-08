#!/usr/bin/env python3
"""Download / inventory DeepShip (cross-domain validation).

Official: https://github.com/irfankamboh/DeepShip
GitHub hosts only a PARTIAL dataset. Full corpus: email mirfan@mail.nwpu.edu.cn
Do NOT scrape third-party mirrors. Never claim complete if only partial.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.download._common import project_data, write_json
from src.data.manifest import rows_to_dataframe, write_manifest
from src.utils.hashing import sha256_file
from src.utils.paths import ensure_dir

GITHUB_REPO = "https://github.com/irfankamboh/DeepShip.git"
AUTHOR_EMAIL = "mirfan@mail.nwpu.edu.cn"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}


def clone_or_update_repo(dest: Path) -> dict:
    ensure_dir(dest.parent)
    meta = {"repo": GITHUB_REPO, "path": str(dest), "ok": False, "error": None}
    try:
        if (dest / ".git").exists():
            subprocess.run(
                ["git", "-C", str(dest), "pull", "--ff-only"],
                check=False,
                capture_output=True,
                text=True,
            )
        elif dest.exists() and any(dest.iterdir()):
            meta["error"] = "destination exists but is not a git clone"
            return meta
        else:
            ensure_dir(dest)
            r = subprocess.run(
                ["git", "clone", "--depth", "1", GITHUB_REPO, str(dest)],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                meta["error"] = r.stderr.strip() or r.stdout.strip()
                return meta
        meta["ok"] = True
    except FileNotFoundError:
        meta["error"] = "git executable not found"
    except Exception as e:  # noqa: BLE001
        meta["error"] = str(e)
    return meta


def classify_from_path(rel: Path) -> tuple[str | None, str | None]:
    parts = [p for p in rel.parts]
    ship_type = None
    ship_id = None
    for p in parts:
        low = p.lower()
        if low in {"cargo", "passenger", "tanker", "tug"}:
            ship_type = p
        if low.startswith(("cargo", "passenger", "tanker", "tug")) and p != ship_type:
            ship_id = p
    return ship_type, ship_id


def build_manifest(raw_dir: Path) -> list[dict]:
    rows = []
    if not raw_dir.exists():
        return rows
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        rel = p.relative_to(raw_dir)
        ship_type, ship_id = classify_from_path(rel)
        rows.append(
            {
                "dataset": "DeepShip",
                "file_path": str(p.as_posix()),
                "recording_id": p.stem,
                "ship_id": ship_id,
                "mmsi": None,
                "ship_type": ship_type,
                "duration_seconds": None,
                "sample_rate": None,
                "channels": None,
                "source_location": "Strait of Georgia (per DeepShip paper)",
                "recording_date": None,
                "session_id": None,
                "distance": None,
                "audibility": None,
                "target_count": None,
                "is_background": False,
                "original_filename": p.name,
                "sha256": sha256_file(p),
                "file_size_bytes": p.stat().st_size,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepShip download / inventory")
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Do not clone GitHub; only inventory data/raw/deepship",
    )
    parser.add_argument(
        "--extra-dir",
        type=str,
        default=None,
        help="Optional local directory with additional DeepShip audio to copy/link inventory",
    )
    args = parser.parse_args()

    raw_dir = project_data("raw", "deepship")
    ensure_dir(raw_dir)
    github_mirror = raw_dir / "_github_partial"
    clone_info = {"skipped": True}
    if not args.skip_clone:
        clone_info = clone_or_update_repo(github_mirror)
        if clone_info.get("ok"):
            # Copy audio from github mirror into class-aware layout without deleting originals
            for p in github_mirror.rglob("*"):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                    rel = p.relative_to(github_mirror)
                    dest = raw_dir / rel
                    ensure_dir(dest.parent)
                    if not dest.exists():
                        dest.write_bytes(p.read_bytes())

    if args.extra_dir:
        extra = Path(args.extra_dir)
        if extra.exists():
            for p in extra.rglob("*"):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                    dest = raw_dir / "from_extra" / p.name
                    ensure_dir(dest.parent)
                    if not dest.exists():
                        dest.write_bytes(p.read_bytes())
        else:
            print(f"WARNING: --extra-dir not found: {extra}")

    rows = build_manifest(raw_dir)
    # Exclude files still under _github_partial from duplicate counting in primary tree
    rows = [r for r in rows if "/_github_partial/" not in r["file_path"].replace("\\", "/")]
    # Re-scan including github for reporting
    all_audio = list(raw_dir.rglob("*"))
    audio_files = [p for p in all_audio if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    classes_found = sorted(
        {
            classify_from_path(p.relative_to(raw_dir))[0]
            for p in audio_files
            if classify_from_path(p.relative_to(raw_dir))[0]
        }
    )
    expected_classes = {"Cargo", "Passenger", "Tanker", "Tug", "cargo", "passenger", "tanker", "tug"}
    missing_classes = sorted(
        c for c in ["Cargo", "Passenger", "Tanker", "Tug"] if c not in classes_found and c.lower() not in {x.lower() for x in classes_found}
    )

    # DeepShip paper: ~47h, 265 ships, 4 classes — GitHub alone is incomplete
    complete = False
    completeness = "not_downloaded" if not audio_files else "partial"

    manifest_path = project_data("metadata", "deepship_manifest.csv")
    df = rows_to_dataframe(rows)
    write_manifest(df, manifest_path)

    status = {
        "dataset": "DeepShip",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "official_repo": GITHUB_REPO,
        "author_email_for_full_dataset": AUTHOR_EMAIL,
        "github_clone": clone_info,
        "audio_file_count": len(audio_files),
        "classes_found": classes_found,
        "missing_classes": missing_classes,
        "complete": complete,
        "completeness": completeness,
        "notes": (
            "GitHub hosts only part of DeepShip. Request the remaining data from "
            f"{AUTHOR_EMAIL}. Do not use unofficial third-party mirrors."
        ),
        "manifest": str(manifest_path),
    }
    write_json(project_data("metadata", "deepship_download_status.json"), status)

    print(json.dumps({k: status[k] for k in status if k != "github_clone"}, indent=2))
    if not clone_info.get("ok") and not args.skip_clone:
        print(f"Clone issue: {clone_info.get('error')}")
    print(f"Manifest: {manifest_path} ({len(rows)} primary rows)")
    print("Complete: False (partial unless author-provided full corpus is present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
