"""Tiny synthetic tests for E1c certified stats/ASP pooling (no training)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.certified_verify import (  # noqa: E402
    _assert_enroll_probe_disjoint,
    _split_ships,
    certified_mean_eval,
    stats_set_eval,
)


def _synth(n_ships=4, n_clips=40, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n_ships * n_clips, dim)).astype(np.float32)
    ids = np.repeat([f"s{i}" for i in range(n_ships)], n_clips)
    return E, ids


def test_stats_and_mean_same_split():
    E, ids = _synth()
    r_mean = certified_mean_eval(E, ids, k_enroll=5, probe_w=3, seed=0)
    r_stats = stats_set_eval(E, ids, k_enroll=5, probe_w=3, seed=0)
    assert r_mean["n_probe"] == r_stats["n_probe"]
    assert r_mean["n_ships"] == r_stats["n_ships"]
    assert r_mean["disjointness_pass"] is True
    assert r_stats["disjointness_pass"] is True
    assert r_stats["protocol"] == "certified_stats_pool"
    assert r_mean["protocol"] == "certified_mean"
    assert r_stats["dim"] == 16  # 2 * 8

    splits_a = _split_ships(np.asarray(ids).astype(str), 5, 3, 0)
    splits_b = _split_ships(np.asarray(ids).astype(str), 5, 3, 0)
    assert splits_a.keys() == splits_b.keys()
    for s in splits_a:
        assert np.array_equal(splits_a[s]["enroll"], splits_b[s]["enroll"])
        assert len(splits_a[s]["probe_chunks"]) == len(splits_b[s]["probe_chunks"])


def test_forced_overlap_raises():
    try:
        _assert_enroll_probe_disjoint("sX", np.array([0, 1, 2]), [2, 3, 4])
        raise AssertionError("expected RuntimeError on enroll∩probe overlap")
    except RuntimeError as e:
        assert "DISJOINTNESS FAIL" in str(e)
        assert "ship=sX" in str(e)


def test_split_ships_disjoint_on_synth():
    E, ids = _synth()
    splits = _split_ships(np.asarray(ids).astype(str), 5, 3, 0)
    for s, sp in splits.items():
        probe_idxs = [i for chunk in sp["probe_chunks"] for i in chunk.tolist()]
        _assert_enroll_probe_disjoint(s, sp["enroll"], probe_idxs)


if __name__ == "__main__":
    test_stats_and_mean_same_split()
    test_forced_overlap_raises()
    test_split_ships_disjoint_on_synth()
    print("ok")
