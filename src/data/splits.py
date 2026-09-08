"""Leakage-controlled split builders for VTUAD (definitions only — no training)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.vtuad_metadata import load_vtuad_metadata
from src.utils.paths import repo_root

SPLIT_COLS = [
    "file_path",
    "protocol",
    "split",
    "MMSI",
    "sub_init",
    "date",
    "label",
    "scenario",
    "distance_band",
    "file_index",
    "sha256",
    "is_background",
]


def load_splits_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else repo_root() / "configs" / "splits.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_catalog(
    raw_root: Path | None = None,
    quality_csv: Path | None = None,
) -> pd.DataFrame:
    """Load VTUAD metadata and optionally join SHA256 from quality report."""
    root = repo_root()
    raw_root = Path(raw_root) if raw_root else root / "data" / "raw" / "vtuad"
    df = load_vtuad_metadata(raw_root)
    if df.empty:
        return df
    df = df.copy()
    df["MMSI"] = df["MMSI"].fillna(0).astype("Int64")
    df["is_background"] = (df["MMSI"].astype(int) == 0) | (df["label"] == "background")
    df["recording_id"] = (
        df["scenario"].astype(str)
        + "|"
        + df["split"].astype(str)
        + "|"
        + df["label"].astype(str)
        + "|"
        + df["file_index"].astype(int).astype(str)
    )
    # Official IEEE folder split name — keep separately
    df["official_split"] = df["split"]

    def _norm_path(p: str) -> str:
        s = str(p).replace("\\", "/").lower()
        # Join on path suffix under data/raw/ so absolute vs relative both match
        marker = "data/raw/"
        if marker in s:
            s = s[s.index(marker) :]
        return s

    qpath = Path(quality_csv) if quality_csv else root / "data" / "metadata" / "data_quality_report.csv"
    if qpath.exists():
        q = pd.read_csv(qpath)
        q["file_path_norm"] = q["file_path"].map(_norm_path)
        df["file_path_norm"] = df["file_path"].map(_norm_path)
        keep = [c for c in ["file_path_norm", "sha256", "duplicate_of"] if c in q.columns]
        df = df.merge(q[keep], on="file_path_norm", how="left")
    else:
        df["sha256"] = None
        df["duplicate_of"] = None
    return df


def _assign_groups(
    groups: list[Any],
    ratios: dict[str, float],
    seed: int,
) -> dict[Any, str]:
    """Deterministic 80/10/10 assignment of groups to train/val/test."""
    rng = np.random.default_rng(seed)
    g = np.array(groups, dtype=object)
    rng.shuffle(g)
    n = len(g)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))
    # remainder → test to guarantee all groups assigned
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1) if n > 1 else 0
        n_train = max(0, n - n_val - (1 if n > n_val else 0))
    n_test = n - n_train - n_val
    assign: dict[Any, str] = {}
    for i, gid in enumerate(g):
        if i < n_train:
            assign[gid] = "train"
        elif i < n_train + n_val:
            assign[gid] = "val"
        else:
            assign[gid] = "test"
    # Ensure at least one in each if possible
    if n >= 3:
        for split, need in [("train", n_train), ("val", n_val), ("test", n_test)]:
            if need == 0 and split not in assign.values():
                # steal from train
                for k, v in list(assign.items()):
                    if v == "train":
                        assign[k] = split
                        break
    return assign


def _finalize(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    out = df.copy()
    out["protocol"] = protocol
    for c in SPLIT_COLS:
        if c not in out.columns:
            out[c] = None
    return out[SPLIT_COLS].reset_index(drop=True)


def split_random(df: pd.DataFrame, seed: int = 42, ratios: dict | None = None) -> pd.DataFrame:
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))
    labels = np.empty(n, dtype=object)
    labels[:n_train] = "train"
    labels[n_train : n_train + n_val] = "val"
    labels[n_train + n_val :] = "test"
    out = df.iloc[idx].copy()
    out["split"] = labels
    return _finalize(out, "random")


def split_mmsi(df: pd.DataFrame, seed: int = 42, ratios: dict | None = None) -> pd.DataFrame:
    """Nonzero MMSI in exactly one split; background by random clip."""
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    ships = df[~df["is_background"]].copy()
    bg = df[df["is_background"]].copy()

    mmsis = sorted(ships["MMSI"].astype(int).unique().tolist())
    assign = _assign_groups(mmsis, ratios, seed)
    ships = ships.copy()
    ships["split"] = ships["MMSI"].astype(int).map(assign)

    bg_split = split_random(bg, seed=seed + 1, ratios=ratios)
    # restore columns from bg for concat
    bg_out = bg.copy()
    bg_out["split"] = bg_split["split"].values

    out = pd.concat([ships, bg_out], ignore_index=True)
    return _finalize(out, "mmsi")


def split_session(df: pd.DataFrame, seed: int = 42, ratios: dict | None = None) -> pd.DataFrame:
    """Each sub_init in exactly one split; prefer ship balance."""
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    # Prefer sessions with more unique ships first for balance
    sess_stats = (
        df.groupby("sub_init")
        .agg(n_ships=("MMSI", lambda s: s[s.fillna(0).astype(int) != 0].nunique()), n_clips=("file_path", "size"))
        .reset_index()
        .sort_values(["n_ships", "n_clips"], ascending=False)
    )
    sessions = sess_stats["sub_init"].tolist()
    # Stratify-ish: shuffle within ship-count buckets then assign
    rng = np.random.default_rng(seed)
    sessions = list(sessions)
    # Re-shuffle while keeping approximate balance via _assign_groups on shuffled list
    rng.shuffle(sessions)
    assign = _assign_groups(sessions, ratios, seed)
    out = df.copy()
    out["split"] = out["sub_init"].map(assign)
    return _finalize(out, "session")


def split_temporal(df: pd.DataFrame, ratios: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Chronological by unique dates — no shuffle."""
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    dates = sorted(df["date"].dropna().unique().tolist())
    if len(dates) < 3:
        raise ValueError(f"Need >=3 unique dates for temporal split; found {len(dates)}")
    n = len(dates)
    n_train = max(1, int(round(n * ratios["train"])))
    n_val = max(1, int(round(n * ratios["val"])))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
        n_train = n - n_val - 1
    train_dates = set(dates[:n_train])
    val_dates = set(dates[n_train : n_train + n_val])
    test_dates = set(dates[n_train + n_val :])
    boundaries = {
        "train_start": int(min(train_dates)),
        "train_end": int(max(train_dates)),
        "val_start": int(min(val_dates)),
        "val_end": int(max(val_dates)),
        "test_start": int(min(test_dates)),
        "test_end": int(max(test_dates)),
        "n_train_dates": len(train_dates),
        "n_val_dates": len(val_dates),
        "n_test_dates": len(test_dates),
    }

    def which(d):
        if d in train_dates:
            return "train"
        if d in val_dates:
            return "val"
        return "test"

    out = df.copy()
    out["split"] = out["date"].map(which)
    return _finalize(out, "temporal"), boundaries


def split_scenario(
    df: pd.DataFrame,
    train_scenario: str,
    val_scenario: str,
    test_scenario: str,
    dedupe_by_sha256: bool = True,
) -> pd.DataFrame:
    """Cross-scenario protocol; optional SHA256 dedupe so content appears once."""
    out = df.copy()
    mapping = {
        train_scenario: "train",
        val_scenario: "val",
        test_scenario: "test",
    }
    out = out[out["scenario"].isin(mapping)].copy()
    out["split"] = out["scenario"].map(mapping)

    if dedupe_by_sha256 and out["sha256"].notna().any():
        # Prefer keeping the earliest distance band (train > val > test priority)
        priority = {"train": 0, "val": 1, "test": 2}
        out["_prio"] = out["split"].map(priority)
        out = out.sort_values(["sha256", "_prio"], kind="mergesort")
        # Keep first occurrence of each sha256; also keep rows with null sha256
        mask_keep = out["sha256"].isna() | ~out.duplicated(subset=["sha256"], keep="first")
        out = out[mask_keep].drop(columns=["_prio"])
    return _finalize(out, "scenario")


def write_protocol_csvs(df: pd.DataFrame, protocol: str, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train", "val", "test"):
        path = out_dir / f"{protocol}_{split}.csv"
        part = df[df["split"] == split]
        part.to_csv(path, index=False)
        paths[split] = path
    return paths


def build_all_protocols(cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_splits_config()
    seed = int(cfg.get("seed", 42))
    ratios = cfg.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    out_dir = repo_root() / cfg.get("output_dir", "data/processed/splits")
    catalog = prepare_catalog(
        quality_csv=repo_root() / cfg.get("quality_report", "data/metadata/data_quality_report.csv")
    )
    if catalog.empty:
        raise RuntimeError("Empty VTUAD catalog")

    results: dict[str, Any] = {"catalog_rows": len(catalog), "protocols": {}}

    # random
    rnd = split_random(catalog, seed=seed, ratios=ratios)
    write_protocol_csvs(rnd, "random", out_dir)
    results["protocols"]["random"] = rnd

    # mmsi
    mmsi = split_mmsi(catalog, seed=seed, ratios=ratios)
    write_protocol_csvs(mmsi, "mmsi", out_dir)
    results["protocols"]["mmsi"] = mmsi

    # session
    sess = split_session(catalog, seed=seed, ratios=ratios)
    write_protocol_csvs(sess, "session", out_dir)
    results["protocols"]["session"] = sess

    # temporal
    temporal, boundaries = split_temporal(catalog, ratios=ratios)
    write_protocol_csvs(temporal, "temporal", out_dir)
    results["protocols"]["temporal"] = temporal
    results["temporal_boundaries"] = boundaries

    # scenario
    scfg = cfg.get("protocols", {}).get("scenario", {})
    scenario = split_scenario(
        catalog,
        train_scenario=scfg.get("train_scenario", "inclusion_2000_exclusion_4000"),
        val_scenario=scfg.get("val_scenario", "inclusion_3000_exclusion_5000"),
        test_scenario=scfg.get("test_scenario", "inclusion_4000_exclusion_6000"),
        dedupe_by_sha256=bool(scfg.get("dedupe_by_sha256", True)),
    )
    write_protocol_csvs(scenario, "scenario", out_dir)
    results["protocols"]["scenario"] = scenario

    # Also save official IEEE folder split as reference baseline
    official = catalog.copy()
    official["split"] = official["official_split"].replace({"validation": "val"})
    official = _finalize(official, "official")
    write_protocol_csvs(official, "official", out_dir)
    results["protocols"]["official"] = official

    return results
