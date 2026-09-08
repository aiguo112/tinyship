"""Tiny synthetic tests for E1b temporal aggregators + aggregator_set_eval."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.certified_verify import (  # noqa: E402
    aggregator_set_eval, certified_mean_eval,
)
from src.models.fingerprint import build_temporal_aggregator  # noqa: E402


def _synth(n_ships=4, n_clips=40, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n_ships * n_clips, dim)).astype(np.float32)
    ids = np.repeat([f"s{i}" for i in range(n_ships)], n_clips)
    sess = np.array(
        [f"sess{i % 3}" for i in range(n_ships * n_clips)], dtype=object,
    )
    return E, ids, sess


def test_forward_shapes():
    import torch

    B, W, D = 2, 5, 8
    H = torch.randn(B, W, D)
    for name in ("gru", "tcn", "mha"):
        mod = build_temporal_aggregator(name, D)
        out = mod(H)
        assert out.shape == (B, D), f"{name} got {out.shape}"
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        # also accept (W, D)
        out2 = mod(H[0])
        assert out2.shape == (1, D)


def test_aggregator_set_eval_n_probe_match():
    import torch

    E, ids, sess = _synth()
    r_mean = certified_mean_eval(E, ids, k_enroll=5, probe_w=3, seed=0)
    for name in ("gru", "tcn", "mha"):
        mod = build_temporal_aggregator(name, 8)
        mod.eval()
        r = aggregator_set_eval(
            E, ids, mod, device="cpu",
            k_enroll=5, probe_w=3, seed=0, sessions=sess,
        )
        assert r["n_probe"] == r_mean["n_probe"], name
        assert r["n_ships"] == r_mean["n_ships"], name
        assert r["disjointness_pass"] is True
        assert r["protocol"] == "certified_temporal_agg"
        assert "session_disjoint_pass" in r
        assert "session_overlap_ships" in r


def test_factory_unknown():
    try:
        build_temporal_aggregator("nope", 8)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "unknown" in str(e).lower()


if __name__ == "__main__":
    test_forward_shapes()
    test_aggregator_set_eval_n_probe_match()
    test_factory_unknown()
    print("ok")
