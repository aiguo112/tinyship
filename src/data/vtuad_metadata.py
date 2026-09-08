"""Load and analyse VTUAD metadata CSVs (IEEE DataPort layout)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

SCENARIOS = (
    "inclusion_2000_exclusion_4000",
    "inclusion_3000_exclusion_5000",
    "inclusion_4000_exclusion_6000",
)
SPLITS = ("train", "validation", "test")
LABELS = ("cargo", "tanker", "tug", "passenger", "background")


def scenario_distance_band(scenario: str) -> str:
    """Human-readable distance band from scenario folder name."""
    if "2000" in scenario and "4000" in scenario:
        return "2-4 km"
    if "3000" in scenario and "5000" in scenario:
        return "3-5 km"
    if "4000" in scenario and "6000" in scenario:
        return "4-6 km"
    return scenario


def load_scenario_split_metadata(scenario_dir: Path, split: str) -> pd.DataFrame:
    """Load one metadata.csv and attach scenario/split columns."""
    meta_path = scenario_dir / split / "metadata.csv"
    if not meta_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(meta_path, index_col=0)
    df = df.reset_index().rename(columns={"index": "row_id"})
    df["scenario"] = scenario_dir.name
    df["distance_band"] = scenario_distance_band(scenario_dir.name)
    df["split"] = split
    df["wav_stem"] = df["file_index"].astype(int).astype(str)
    df["file_path"] = df.apply(
        lambda r: str(
            scenario_dir
            / split
            / "audio"
            / str(r["label"]).lower()
            / f"{int(r['file_index'])}.wav"
        ),
        axis=1,
    )
    return df


def load_vtuad_metadata(
    raw_root: str | Path,
    scenarios: Iterable[str] | None = None,
    splits: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Concatenate all VTUAD metadata.csv files under raw_root."""
    raw_root = Path(raw_root)
    scenarios = tuple(scenarios or SCENARIOS)
    splits = tuple(splits or SPLITS)
    parts: list[pd.DataFrame] = []
    for scen in scenarios:
        scen_dir = raw_root / scen
        if not scen_dir.is_dir():
            continue
        for split in splits:
            part = load_scenario_split_metadata(scen_dir, split)
            if not part.empty:
                parts.append(part)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["MMSI"] = out["MMSI"].astype("Int64")
    out["date"] = out["date"].astype("Int64")
    out["session_id"] = out["sub_init"].astype("Int64")
    out["ship_id"] = out["MMSI"]
    out["ship_type"] = out["label"]
    return out


def mmsi_split_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """For each MMSI, list which official splits it appears in (per scenario)."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (scenario, mmsi), g in df.groupby(["scenario", "MMSI"], dropna=True):
        if pd.isna(mmsi):
            continue
        splits = sorted(g["split"].unique().tolist())
        rows.append(
            {
                "scenario": scenario,
                "MMSI": int(mmsi),
                "splits": ",".join(splits),
                "n_splits": len(splits),
                "n_rows": len(g),
                "labels": ",".join(sorted(g["label"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def cross_scenario_mmsi(df: pd.DataFrame) -> pd.DataFrame:
    """MMSI appearing in more than one distance scenario."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for mmsi, g in df.groupby("MMSI", dropna=True):
        if pd.isna(mmsi):
            continue
        scenarios = sorted(g["scenario"].unique().tolist())
        if len(scenarios) <= 1:
            continue
        rows.append(
            {
                "MMSI": int(mmsi),
                "n_scenarios": len(scenarios),
                "scenarios": ",".join(scenarios),
                "n_rows": len(g),
            }
        )
    return pd.DataFrame(rows)


def join_quality_report(
    meta: pd.DataFrame,
    quality_csv: str | Path,
) -> pd.DataFrame:
    """Attach sha256 / duplicate_of from data_quality_report.csv."""
    q = pd.read_csv(quality_csv)
    q["file_path_norm"] = q["file_path"].str.replace("\\", "/", regex=False).str.lower()
    meta = meta.copy()
    meta["file_path_norm"] = meta["file_path"].str.replace("\\", "/", regex=False).str.lower()
    cols = ["file_path_norm", "sha256", "duplicate_of", "duration_seconds", "sample_rate"]
    cols = [c for c in cols if c in q.columns]
    return meta.merge(q[cols], on="file_path_norm", how="left")


def samples_per_mmsi(df: pd.DataFrame) -> pd.Series:
    """Count metadata rows per MMSI (non-background only by default)."""
    ship = df[df["label"] != "background"].copy()
    return ship.groupby("MMSI").size().sort_values(ascending=False)
