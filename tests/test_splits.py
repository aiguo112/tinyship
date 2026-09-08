"""Tests for leakage-controlled splits (no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import prepare_catalog, split_mmsi, split_random, split_session


def test_mmsi_disjoint():
    cat = prepare_catalog()
    if cat.empty:
        return
    df = split_mmsi(cat, seed=42)
    sets = {}
    for sp in ("train", "val", "test"):
        part = df[df["split"] == sp]
        sets[sp] = set(part["MMSI"].dropna().astype(int)) - {0}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
    assert len(sets["train"]) + len(sets["val"]) + len(sets["test"]) == 222


def test_session_disjoint():
    cat = prepare_catalog()
    if cat.empty:
        return
    df = split_session(cat, seed=42)
    sets = {}
    for sp in ("train", "val", "test"):
        sets[sp] = set(df.loc[df["split"] == sp, "sub_init"].dropna().astype(int))
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])


def test_random_leaks_ships():
    cat = prepare_catalog()
    if cat.empty:
        return
    df = split_random(cat, seed=42)
    tr = set(df.loc[df.split == "train", "MMSI"].dropna().astype(int)) - {0}
    te = set(df.loc[df.split == "test", "MMSI"].dropna().astype(int)) - {0}
    assert len(tr & te) > 0
