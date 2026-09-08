#!/usr/bin/env python3
"""Build leakage-controlled VTUAD split CSVs (no audio copy, no training)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from src.data.splits import (
    build_all_protocols,
    load_splits_config,
    prepare_catalog,
    split_mmsi,
    split_random,
    split_scenario,
    split_session,
    split_temporal,
    write_protocol_csvs,
)
from src.utils.paths import repo_root


def update_config_temporal_boundaries(boundaries: dict) -> None:
    cfg_path = repo_root() / "configs" / "splits.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("protocols", {}).setdefault("temporal", {})["date_boundaries"] = boundaries
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def build_one(protocol: str, cfg: dict) -> None:
    seed = int(cfg.get("seed", 42))
    ratios = cfg.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    out_dir = repo_root() / cfg.get("output_dir", "data/processed/splits")
    catalog = prepare_catalog(
        quality_csv=repo_root() / cfg.get("quality_report", "data/metadata/data_quality_report.csv")
    )
    if protocol == "random":
        df = split_random(catalog, seed=seed, ratios=ratios)
    elif protocol == "mmsi":
        df = split_mmsi(catalog, seed=seed, ratios=ratios)
    elif protocol == "session":
        df = split_session(catalog, seed=seed, ratios=ratios)
    elif protocol == "temporal":
        df, boundaries = split_temporal(catalog, ratios=ratios)
        update_config_temporal_boundaries(boundaries)
        print("Temporal boundaries:", json.dumps(boundaries, indent=2))
    elif protocol == "scenario":
        scfg = cfg.get("protocols", {}).get("scenario", {})
        df = split_scenario(
            catalog,
            train_scenario=scfg.get("train_scenario", "inclusion_2000_exclusion_4000"),
            val_scenario=scfg.get("val_scenario", "inclusion_3000_exclusion_5000"),
            test_scenario=scfg.get("test_scenario", "inclusion_4000_exclusion_6000"),
            dedupe_by_sha256=bool(scfg.get("dedupe_by_sha256", True)),
        )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    paths = write_protocol_csvs(df, protocol, out_dir)
    for k, p in paths.items():
        print(f"  {protocol}_{k}: {len(pd_read(p))} rows -> {p}")


def pd_read(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build VTUAD split definitions")
    parser.add_argument(
        "--protocol",
        choices=["random", "mmsi", "session", "temporal", "scenario", "all"],
        default="all",
    )
    args = parser.parse_args()
    cfg = load_splits_config()

    if args.protocol == "all":
        results = build_all_protocols(cfg)
        if "temporal_boundaries" in results:
            update_config_temporal_boundaries(results["temporal_boundaries"])
            print("Temporal boundaries:", json.dumps(results["temporal_boundaries"], indent=2))
        out_dir = repo_root() / cfg.get("output_dir", "data/processed/splits")
        for name, df in results["protocols"].items():
            print(f"{name}: train={sum(df.split=='train')} val={sum(df.split=='val')} test={sum(df.split=='test')}")
        print(f"Wrote splits under {out_dir}")
    else:
        build_one(args.protocol, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
