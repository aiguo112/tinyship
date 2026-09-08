#!/usr/bin/env python3
"""Freeze VTUAD benchmark artifacts and verify forbidden overlaps.

Does NOT train models. Does NOT modify data/raw/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.utils.paths import repo_root

LOCK_VERSION = "v1.0"

# Artifacts that define the locked benchmark
LOCKED_PATHS = [
    "configs/splits.yaml",
    "data/metadata/vtuad_manifest.csv",
    "data/metadata/vtuad_leakage_report.json",
    "data/metadata/data_quality_report.csv",
    "reports/split_audit.md",
    "reports/split_statistics.csv",
    # Core experimental splits for ShipNN comparison
    "data/processed/splits/random_train.csv",
    "data/processed/splits/random_val.csv",
    "data/processed/splits/random_test.csv",
    "data/processed/splits/session_train.csv",
    "data/processed/splits/session_val.csv",
    "data/processed/splits/session_test.csv",
    "data/processed/splits/mmsi_train.csv",
    "data/processed/splits/mmsi_val.csv",
    "data/processed/splits/mmsi_test.csv",
    "data/processed/splits/temporal_train.csv",
    "data/processed/splits/temporal_val.csv",
    "data/processed/splits/temporal_test.csv",
    "data/processed/splits/scenario_train.csv",
    "data/processed/splits/scenario_val.csv",
    "data/processed/splits/scenario_test.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_split(protocol: str) -> pd.DataFrame:
    d = repo_root() / "data" / "processed" / "splits"
    return pd.concat(
        [pd.read_csv(d / f"{protocol}_{s}.csv") for s in ("train", "val", "test")],
        ignore_index=True,
    )


def verify_forbidden_overlaps() -> dict:
    """Enforce protocol-specific zero-overlap constraints."""
    checks = {}

    # MMSI: nonzero MMSI disjoint
    mmsi = load_split("mmsi")
    sets = {}
    for sp in ("train", "val", "test"):
        part = mmsi[mmsi["split"] == sp]
        sets[sp] = set(part["MMSI"].dropna().astype(int)) - {0}
    checks["mmsi_nonzero_ship_overlap"] = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
        "pass": (
            len(sets["train"] & sets["val"]) == 0
            and len(sets["train"] & sets["test"]) == 0
            and len(sets["val"] & sets["test"]) == 0
        ),
        "n_ships": {k: len(v) for k, v in sets.items()},
    }

    # Session: sub_init disjoint
    sess = load_split("session")
    ssets = {}
    for sp in ("train", "val", "test"):
        ssets[sp] = set(sess.loc[sess["split"] == sp, "sub_init"].dropna().astype(int))
    checks["session_sub_init_overlap"] = {
        "train_val": len(ssets["train"] & ssets["val"]),
        "train_test": len(ssets["train"] & ssets["test"]),
        "val_test": len(ssets["val"] & ssets["test"]),
        "pass": (
            len(ssets["train"] & ssets["val"]) == 0
            and len(ssets["train"] & ssets["test"]) == 0
            and len(ssets["val"] & ssets["test"]) == 0
        ),
        "n_sessions": {k: len(v) for k, v in ssets.items()},
    }

    # File-path uniqueness within each protocol (no clip in two splits)
    for protocol in ("random", "session", "mmsi", "temporal", "scenario"):
        df = load_split(protocol)
        dup = 0
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            fa = set(df.loc[df.split == a, "file_path"].astype(str))
            fb = set(df.loc[df.split == b, "file_path"].astype(str))
            dup += len(fa & fb)
        checks[f"{protocol}_file_path_overlap"] = {"overlap_total": dup, "pass": dup == 0}

    # Random MUST show ship leakage (sanity that baseline is leaky)
    rnd = load_split("random")
    rsets = {}
    for sp in ("train", "val", "test"):
        rsets[sp] = set(rnd.loc[rnd.split == sp, "MMSI"].dropna().astype(int)) - {0}
    leak = len(rsets["train"] & rsets["test"])
    checks["random_expected_mmsi_leakage"] = {
        "train_test_overlap": leak,
        "pass": leak > 0,
        "note": "Random baseline must leak ships; otherwise comparison is meaningless.",
    }

    all_pass = all(v.get("pass", False) for v in checks.values())
    return {"all_pass": all_pass, "checks": checks}


def main() -> int:
    root = repo_root()
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    files = []
    missing = []
    for rel in LOCKED_PATHS:
        path = root / rel
        if not path.exists():
            missing.append(rel)
            continue
        files.append(
            {
                "path": rel.replace("\\", "/"),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    overlaps = verify_forbidden_overlaps()

    lock = {
        "lock_version": LOCK_VERSION,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "VTUAD",
        "scientific_roles": {
            "closed_set_vessel_identification_primary": "session",
            "unseen_vessel_generalization_primary": "mmsi",
            "shipnn_leakage_baseline": "random",
        },
        "do_not_train_fingerprint_model_yet": True,
        "missing_files": missing,
        "files": files,
        "overlap_verification": overlaps,
        "next_milestone": [
            "E0 ShipNN on random",
            "E1 ShipNN on session (closed-set primary)",
            "E2 ShipNN on MMSI (unseen-vessel; interpret carefully)",
            "Produce accuracy comparison table",
            "Only then design TinyShip-Fingerprint",
        ],
    }

    lock_path = reports / "benchmark_lock_manifest.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    # Human-readable lock doc
    lines = [
        f"# Benchmark Lock — {LOCK_VERSION}",
        "",
        f"Locked at (UTC): `{lock['locked_at_utc']}`",
        "",
        "## Status",
        "",
        f"- Overlap verification: **{'PASS' if overlaps['all_pass'] else 'FAIL'}**",
        "- Models trained: **NO**",
        "- Fingerprint network: **NOT started** (by design)",
        "",
        "## Scientific protocol roles (locked)",
        "",
        "| Task | Primary split | Why |",
        "|---|---|---|",
        "| Closed-set vessel identification | **session** | Same ship, different session/day is realistic ID; blocks clip/session leakage |",
        "| Unseen-vessel / fingerprinting generalization | **mmsi** | Disjoint identities; needs enrollment/retrieval later, not softmax on unseen classes |",
        "| ShipNN leakage baseline | **random** | Reproduce reported setting and expose inflation |",
        "",
        "## Forbidden-overlap checks",
        "",
        "```json",
        json.dumps(overlaps["checks"], indent=2),
        "```",
        "",
        "## Locked files",
        "",
        f"See SHA256 list in `{lock_path.name}` ({len(files)} files).",
        "",
        "## Rules after lock",
        "",
        "1. Do **not** regenerate `data/processed/splits/*` without bumping `lock_version`.",
        "2. Do **not** modify `data/raw/`.",
        "3. Do **not** overwrite prior metadata reports casually.",
        "4. Next code milestone: **ShipNN reproduction on random / session / MMSI only**.",
        "5. Do **not** implement TinyShip-Fingerprint until the ShipNN comparison table exists.",
        "",
    ]
    (reports / "BENCHMARK_LOCK.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {lock_path}")
    print(f"Wrote {reports / 'BENCHMARK_LOCK.md'}")
    print(f"Overlap verification: {'PASS' if overlaps['all_pass'] else 'FAIL'}")
    if missing:
        print("MISSING:", missing)
        return 1
    return 0 if overlaps["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
