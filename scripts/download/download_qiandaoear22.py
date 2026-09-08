#!/usr/bin/env python3
"""Download / register QiandaoEar22.

Official GitHub: https://github.com/xiaoyangdu22/QiandaoEar22
External hosting: Microsoft OneDrive (link in README) and IEEE DataPort.
Do not invent or bypass access restrictions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.download._common import print_missing_credentials, project_data, write_json
from src.data.manifest import rows_to_dataframe, write_manifest
from src.utils.hashing import sha256_file
from src.utils.paths import ensure_dir

GITHUB = "https://github.com/xiaoyangdu22/QiandaoEar22"
IEEE = "https://ieee-dataport.org/documents/qiandaoear22"
ONEDRIVE_README_URL = (
    "https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/"
    "duxiaoyang22_mails_ucas_ac_cn/EomiGNu7mO5FmUke62y6Q7IBIP64kpJrJMJOZp_c-qkFAA?e=x8gxuL"
)
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz"}


def parse_metadata_from_name(name: str) -> dict:
    """Best-effort parse of QiandaoEar22 naming; leave unknowns as None."""
    low = name.lower()
    is_bg = "noise" in low or low.startswith("data0")
    target_count = None
    if "multi" in low:
        target_count = "multiple"
    elif "single" in low or "target" in low:
        target_count = "single"
    distance = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*m", low)
    if m:
        distance = m.group(1)
    ship_id = None
    for key in ("kaiyuan", "uuv", "speedboat", "speed_boat"):
        if key in low:
            ship_id = key
            break
    return {
        "is_background": is_bg,
        "target_count": target_count,
        "distance": distance,
        "ship_id": ship_id,
        "audibility": None,
    }


def inventory_archives(raw_dir: Path) -> list[dict]:
    out = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in ARCHIVE_EXTS:
            out.append(
                {
                    "path": str(p),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )
    return out


def extract_archive(archive: Path, dest: Path) -> None:
    ensure_dir(dest)
    suf = archive.suffix.lower()
    if suf == ".zip":
        import zipfile

        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif suf in {".tar", ".gz"} or archive.name.endswith(".tar.gz"):
        import tarfile

        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)
    else:
        print(
            f"Archive type {archive.suffix} requires manual extraction "
            f"(e.g. 7-Zip for .rar): {archive}"
        )


def build_manifest(raw_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        meta = parse_metadata_from_name(p.name)
        rows.append(
            {
                "dataset": "QiandaoEar22",
                "file_path": str(p.as_posix()),
                "recording_id": p.stem,
                "ship_id": meta["ship_id"],
                "mmsi": None,
                "ship_type": meta["ship_id"],
                "duration_seconds": None,
                "sample_rate": None,
                "channels": None,
                "source_location": "Qiandao Lake, China",
                "recording_date": "2022-06",
                "session_id": None,
                "distance": meta["distance"],
                "audibility": meta["audibility"],
                "target_count": meta["target_count"],
                "is_background": meta["is_background"],
                "original_filename": p.name,
                "sha256": sha256_file(p),
                "file_size_bytes": p.stat().st_size,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="QiandaoEar22 download / inventory")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract zip/tar archives found under data/raw/qiandaoear22/",
    )
    parser.add_argument(
        "--try-onedrive",
        action="store_true",
        help="Attempt direct HTTP fetch of the README OneDrive URL (often fails without browser auth)",
    )
    args = parser.parse_args()

    raw_dir = project_data("raw", "qiandaoear22")
    ensure_dir(raw_dir)

    onedrive_attempt = None
    if args.try_onedrive:
        from scripts.download._common import download_file

        dest = raw_dir / "onedrive_fetch.bin"
        try:
            onedrive_attempt = download_file(ONEDRIVE_README_URL, dest)
            # SharePoint folder links typically return HTML, not a dataset archive
            head = dest.read_bytes()[:200].lower()
            if b"<html" in head or b"sharepoint" in head:
                onedrive_attempt["complete"] = False
                onedrive_attempt["note"] = (
                    "URL returned HTML/SharePoint page, not a dataset archive. "
                    "Manual browser download required."
                )
                dest.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            onedrive_attempt = {"ok": False, "error": str(e)}

    archives = inventory_archives(raw_dir)
    if args.extract:
        for a in archives:
            extract_archive(Path(a["path"]), raw_dir)

    rows = build_manifest(raw_dir)
    manifest_path = project_data("metadata", "qiandaoear22_manifest.csv")
    write_manifest(rows_to_dataframe(rows), manifest_path)

    audio_count = len(rows)
    complete = False
    completeness = "not_downloaded" if audio_count == 0 and not archives else (
        "partial_or_unknown" if audio_count > 0 or archives else "not_downloaded"
    )

    status = {
        "dataset": "QiandaoEar22",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "official_github": GITHUB,
        "ieee_dataport": IEEE,
        "onedrive_readme_url": ONEDRIVE_README_URL,
        "onedrive_attempt": onedrive_attempt,
        "archives_found": archives,
        "audio_file_count": audio_count,
        "complete": complete,
        "completeness": completeness,
        "manifest": str(manifest_path),
        "manual_instructions": [
            f"Open GitHub README: {GITHUB}",
            f"Download dataset folder from OneDrive: {ONEDRIVE_README_URL}",
            f"Or IEEE DataPort (may require subscription): {IEEE}",
            "Place archives or extracted WAV files under data/raw/qiandaoear22/",
            "Re-run: python scripts/download/download_qiandaoear22.py --extract",
        ],
    }
    write_json(project_data("metadata", "qiandaoear22_download_status.json"), status)

    if audio_count == 0 and not archives:
        print_missing_credentials(
            ["Local QiandaoEar22 archives/audio under data/raw/qiandaoear22/"],
            status["manual_instructions"],
        )
    print(json.dumps({k: status[k] for k in ("audio_file_count", "completeness", "complete", "archives_found")}, indent=2, default=str))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
