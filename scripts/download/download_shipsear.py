#!/usr/bin/env python3
"""Download / register ShipsEar.

Official site: https://atlanttic.uvigo.es/underwaternoise/
Public sample WAVs may be available; full database requires email request.
Do NOT use third-party framed train/test splits as the scientific benchmark.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.download._common import download_file, print_missing_credentials, project_data, write_json
from src.data.manifest import rows_to_dataframe, write_manifest
from src.utils.hashing import sha256_file
from src.utils.paths import ensure_dir

OFFICIAL = "https://atlanttic.uvigo.es/underwaternoise/"
AUDIO_EXTS = {".wav", ".flac", ".mp3"}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


def absolutize(base: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        from urllib.parse import urlparse

        u = urlparse(base)
        return f"{u.scheme}://{u.netloc}{href}"
    if not base.endswith("/"):
        base = base.rsplit("/", 1)[0] + "/"
    return base + href


def discover_sample_wavs(page_url: str = OFFICIAL) -> list[str]:
    try:
        import ssl

        ctx = ssl.create_default_context()
        try:
            import certifi

            ctx.load_verify_locations(certifi.where())
        except Exception:
            pass
        with urllib.request.urlopen(page_url, timeout=30, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        print(f"Could not fetch ShipsEar page: {e}")
        print("Manual download from the official site is required if SSL/network fails.")
        return []
    parser = LinkParser()
    parser.feed(html)
    wavs = []
    for href in parser.hrefs:
        if re.search(r"\.wav($|\?)", href, re.I):
            wavs.append(absolutize(page_url, href))
    return sorted(set(wavs))


def build_manifest(raw_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        # ShipsEar sample filenames often start with numeric id
        m = re.match(r"^(\d+)", p.stem)
        recording_id = m.group(1) if m else p.stem
        rows.append(
            {
                "dataset": "ShipsEar",
                "file_path": str(p.as_posix()),
                "recording_id": recording_id,
                "ship_id": None,
                "mmsi": None,
                "ship_type": None,
                "duration_seconds": None,
                "sample_rate": None,
                "channels": None,
                "source_location": None,
                "recording_date": None,
                "session_id": None,
                "distance": None,
                "audibility": None,
                "target_count": None,
                "is_background": "noise" in p.name.lower() or "background" in p.name.lower(),
                "original_filename": p.name,
                "sha256": sha256_file(p),
                "file_size_bytes": p.stat().st_size,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="ShipsEar download / inventory")
    parser.add_argument(
        "--download-samples",
        action="store_true",
        help="Attempt to download publicly linked sample WAV files from the official page",
    )
    args = parser.parse_args()

    raw_dir = project_data("raw", "shipsear")
    ensure_dir(raw_dir)
    samples_dir = raw_dir / "public_samples"
    ensure_dir(samples_dir)

    discovered: list[str] = []
    downloaded = []
    if args.download_samples:
        discovered = discover_sample_wavs()
        for url in discovered:
            name = url.split("?")[0].rstrip("/").split("/")[-1]
            dest = samples_dir / name
            try:
                info = download_file(url, dest)
                downloaded.append(info)
                print(f"Downloaded {name} ({info['bytes']} bytes)")
            except urllib.error.HTTPError as e:
                print(f"HTTP {e.code} for {url}")
            except Exception as e:  # noqa: BLE001
                print(f"Failed {url}: {e}")

    rows = build_manifest(raw_dir)
    manifest_path = project_data("metadata", "shipsear_manifest.csv")
    write_manifest(rows_to_dataframe(rows), manifest_path)

    complete = False
    completeness = "not_downloaded" if not rows else "partial_samples_only"

    status = {
        "dataset": "ShipsEar",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "official_site": OFFICIAL,
        "discovered_sample_urls": discovered,
        "downloaded_samples": downloaded,
        "audio_file_count": len(rows),
        "complete": complete,
        "completeness": completeness,
        "manifest": str(manifest_path),
        "manual_instructions": [
            f"Visit {OFFICIAL}",
            "Download any public sample WAVs from the site",
            "For full database access, send a request to the contact email on that page",
            "Place files under data/raw/shipsear/ (do not use third-party preprocessed splits as benchmark)",
            "Re-run: python scripts/download/download_shipsear.py",
        ],
        "warning": (
            "Third-party repos (e.g. framed 5s splits) note that their train/test "
            "division is not independent — do not use those splits as the scientific benchmark."
        ),
    }
    write_json(project_data("metadata", "shipsear_download_status.json"), status)

    if not rows:
        print_missing_credentials(
            ["Local ShipsEar audio under data/raw/shipsear/"],
            status["manual_instructions"],
        )
    print(json.dumps({k: status[k] for k in ("audio_file_count", "completeness", "complete")}, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
