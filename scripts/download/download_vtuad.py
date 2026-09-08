#!/usr/bin/env python3
"""Download / register VTUAD (ShipNN primary dataset).

Official routes:
  - IEEE DataPort (subscriber): https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data
  - Ocean Networks Canada Oceans 3.0 API (requires ONC_TOKEN)
  - Related pipeline: https://github.com/lucascesarfd/onc_dataset

Does NOT invent credentials. Writes data/metadata/vtuad_download_status.json.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.download._common import print_missing_credentials, project_data, write_json
from src.utils.hashing import sha256_file
from src.utils.paths import ensure_dir


IEEE_URL = "https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data"
ONC_LOGIN = "https://data.oceannetworks.ca/Login"
ONC_API_DOCS = "https://wiki.oceannetworks.ca/display/O2A/Oceans+3.0+API+Home"

SCENARIO_ZIPS = [
    "inclusion_2000_exclusion_4000.zip",
    "inclusion_3000_exclusion_5000.zip",
    "inclusion_4000_exclusion_6000.zip",
]


def inventory_local(raw_dir: Path) -> dict:
    files = []
    total_bytes = 0
    if raw_dir.exists():
        for p in sorted(raw_dir.rglob("*")):
            if p.is_file() and p.name != ".gitkeep":
                size = p.stat().st_size
                total_bytes += size
                files.append(
                    {
                        "path": str(p.relative_to(raw_dir)),
                        "bytes": size,
                        "sha256": sha256_file(p) if size < (2 * 1024**3) else None,
                    }
                )
    return {"file_count": len(files), "total_bytes": total_bytes, "files": files}


def main() -> int:
    parser = argparse.ArgumentParser(description="VTUAD download / status helper")
    parser.add_argument(
        "--local-zip",
        action="append",
        default=[],
        help="Path to an already-downloaded VTUAD zip to register/verify (repeatable)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract registered local zips into data/raw/vtuad/",
    )
    args = parser.parse_args()

    raw_dir = project_data("raw", "vtuad")
    ensure_dir(raw_dir)
    status_path = project_data("metadata", "vtuad_download_status.json")

    onc_token = os.environ.get("ONC_TOKEN", "").strip()
    ieee_user = os.environ.get("IEEE_DATAPORT_USER", "").strip()
    ieee_pass = os.environ.get("IEEE_DATAPORT_PASS", "").strip()
    ieee_token = os.environ.get("IEEE_DATAPORT_TOKEN", "").strip()

    missing: list[str] = []
    if not onc_token:
        missing.append("ONC_TOKEN (Oceans 3.0 API token)")
    if not (ieee_token or (ieee_user and ieee_pass)):
        missing.append(
            "IEEE_DATAPORT_TOKEN or IEEE_DATAPORT_USER + IEEE_DATAPORT_PASS "
            "(IEEE DataPort subscriber credentials)"
        )

    registered = []
    for zp in args.local_zip:
        src = Path(zp)
        if not src.exists():
            print(f"WARNING: local zip not found: {src}")
            continue
        dest = raw_dir / src.name
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
        info = {
            "source": str(src),
            "path": str(dest),
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
        }
        registered.append(info)
        print(f"Registered {dest.name} ({info['bytes']} bytes)")
        if args.extract:
            import zipfile

            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(raw_dir)
            print(f"Extracted {dest.name}")

    local = inventory_local(raw_dir)
    complete = False  # Never claim complete without verified full corpus
    if local["file_count"] > 0:
        completeness = "partial_or_unknown"
    else:
        completeness = "not_downloaded"

    status = {
        "dataset": "VTUAD",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "official_sources": {
            "ieee_dataport": IEEE_URL,
            "onc_login": ONC_LOGIN,
            "onc_api_docs": ONC_API_DOCS,
            "expected_scenario_zips": SCENARIO_ZIPS,
        },
        "credentials": {
            "ONC_TOKEN": bool(onc_token),
            "IEEE_DATAPORT": bool(ieee_token or (ieee_user and ieee_pass)),
        },
        "automated_download_attempted": False,
        "reason_no_auto_download": (
            "IEEE DataPort files require an active subscriber session; "
            "ONC raw reconstruction requires ONC_TOKEN and the onc_dataset pipeline. "
            "Place subscriber zips under data/raw/vtuad/ or pass --local-zip."
            if missing
            else "Credentials present but automated IEEE/ONC bulk download is not "
            "implemented without interactive session cookies; use --local-zip."
        ),
        "registered_local_zips": registered,
        "local_inventory": local,
        "complete": complete,
        "completeness": completeness,
    }
    write_json(status_path, status)

    if missing:
        print_missing_credentials(
            missing,
            [
                f"Create an Oceans 3.0 account and copy your API token: {ONC_LOGIN}",
                f"Set environment variable ONC_TOKEN=<token> ({ONC_API_DOCS})",
                f"Subscribe / log in to IEEE DataPort and download VTUAD zips: {IEEE_URL}",
                "Expected scenario archives: " + ", ".join(SCENARIO_ZIPS),
                "Place zips in data/raw/vtuad/ then re-run: "
                "python scripts/download/download_vtuad.py --local-zip <path> --extract",
            ],
        )
    else:
        print("Credentials detected. Automated bulk download still requires manual "
              "IEEE session or onc_dataset pipeline; use --local-zip for acquired files.")

    print(f"\nStatus written: {status_path}")
    print(f"Local files: {local['file_count']} ({local['total_bytes']} bytes)")
    print(f"Complete: {complete} ({completeness})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
