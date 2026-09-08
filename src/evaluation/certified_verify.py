"""
Certified open-set verification protocol (LOCKED).

Identical to scripts/reproduce/certify_attention_eer.py:
  k_enroll=20 mean prototype, probe_w=3 mean-pool, seed=0,
  per-vessel enroll ∩ probe == ∅ asserted (abort on fail).
"""
from __future__ import annotations
import numpy as np

K_ENROLL = 20
PROBE_W = 3
SEED = 0


def eer(pos: np.ndarray, neg: np.ndarray, rng=None) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    if len(neg) > 50 * len(pos):
        rng = np.random.default_rng(0) if rng is None else rng
        neg = rng.choice(neg, size=50 * len(pos), replace=False)
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    thr = np.quantile(s, np.linspace(0, 1, 1500)) if len(np.unique(s)) > 1500 else np.unique(s)
    best = (1.0, 0.5)
    for t in thr:
        p = s >= t
        far = float(np.mean(p[y == 0])) if (y == 0).any() else 0.0
        frr = float(np.mean(~p[y == 1])) if (y == 1).any() else 0.0
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return float(best[1])


def l2_normalize(E: np.ndarray) -> np.ndarray:
    E = E.astype(np.float64)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def mean_proto(E: np.ndarray, idx: np.ndarray) -> np.ndarray:
    v = E[idx].mean(0)
    return v / (np.linalg.norm(v) + 1e-9)


def _assert_enroll_probe_disjoint(ship, enroll, probe_idxs):
    """Abort if enroll indices overlap probe indices. Used by _split_ships."""
    overlap = set(np.asarray(enroll).tolist()) & set(list(probe_idxs))
    if overlap:
        raise RuntimeError(
            f"DISJOINTNESS FAIL ship={ship} overlap_n={len(overlap)} "
            f"sample={sorted(list(overlap))[:5]}"
        )


def _split_ships(ids: np.ndarray, k_enroll: int, probe_w: int, seed: int):
    ids = np.asarray(ids).astype(str)
    rng = np.random.default_rng(seed)
    usable = [s for s in sorted(np.unique(ids)) if (ids == s).sum() >= k_enroll + probe_w]
    splits = {}
    for s in usable:
        idx = np.where(ids == s)[0].copy()
        rng.shuffle(idx)
        enroll = idx[:k_enroll]
        rest = idx[k_enroll:]
        probe_chunks = []
        probe_idxs = []
        for start in range(0, len(rest) - probe_w + 1, probe_w):
            chunk = rest[start:start + probe_w]
            probe_chunks.append(chunk)
            probe_idxs.extend(chunk.tolist())
        _assert_enroll_probe_disjoint(s, enroll, probe_idxs)
        splits[s] = {"enroll": enroll, "probe_chunks": probe_chunks, "rest": rest}
    return splits


def score_protos(P: np.ndarray, Pid: np.ndarray, probes: list[tuple[np.ndarray, str]]):
    pos, neg = [], []
    for v, sid in probes:
        sims = P @ v
        pos.append(float(sims[Pid == sid][0]))
        neg.extend(sims[Pid != sid].tolist())
    return eer(np.array(pos), np.array(neg)), len(probes)


def top1_protos(P: np.ndarray, Pid: np.ndarray, probes: list[tuple[np.ndarray, str]]) -> float:
    if not probes:
        return float("nan")
    hit = 0
    for v, sid in probes:
        pred = Pid[np.argmax(P @ v)]
        hit += int(pred == sid)
    return hit / len(probes)


def certified_mean_eval(
    E: np.ndarray,
    ids: np.ndarray,
    k_enroll: int = K_ENROLL,
    probe_w: int = PROBE_W,
    seed: int = SEED,
) -> dict:
    """Mean prototype + mean-pooled probe sets. Asserts enroll∩probe=∅."""
    E = l2_normalize(E)
    ids = np.asarray(ids).astype(str)
    splits = _split_ships(ids, k_enroll, max(probe_w, 1), seed)
    if len(splits) < 2:
        raise RuntimeError(f"too few usable ships: {len(splits)}")

    Pid = np.array(list(splits.keys()))
    P = np.stack([mean_proto(E, splits[s]["enroll"]) for s in Pid])
    probes = []
    if probe_w <= 1:
        for s in Pid:
            for i in splits[s]["rest"]:
                probes.append((E[i], s))
    else:
        for s in Pid:
            for chunk in splits[s]["probe_chunks"]:
                probes.append((mean_proto(E, chunk), s))

    e, n = score_protos(P, Pid, probes)
    return {
        "eer": round(e, 4),
        "top1": round(top1_protos(P, Pid, probes), 4),
        "n_ships": len(Pid),
        "n_probe": n,
        "k_enroll": k_enroll,
        "probe_w": probe_w,
        "disjointness_pass": True,
        "protocol": "certified_mean" if probe_w > 1 else "prototype_single_probe",
    }


def clip_cosine_eval(E: np.ndarray, ids: np.ndarray, seed: int = SEED) -> dict:
    """
    Inherent clip-level: random half gallery / half probe per ship.
    EER from ALL same-id probe–gallery cosine pairs vs impostor pairs
    (matches the paper's clip_split protocol, not max-pool genuines).
    """
    E = l2_normalize(E)
    ids = np.asarray(ids).astype(str)
    rng = np.random.default_rng(seed)
    gal_idx, prb_idx = [], []
    for s in np.unique(ids):
        idx = np.where(ids == s)[0].copy()
        if len(idx) < 2:
            continue
        rng.shuffle(idx)
        cut = max(1, len(idx) // 2)
        gal_idx.extend(idx[:cut].tolist())
        prb_idx.extend(idx[cut:].tolist())
    gal_idx = np.asarray(gal_idx)
    prb_idx = np.asarray(prb_idx)
    if set(gal_idx.tolist()) & set(prb_idx.tolist()):
        raise RuntimeError("DISJOINTNESS FAIL clip_cosine gallery∩probe")
    G, Gid = E[gal_idx], ids[gal_idx]
    P, Pid = E[prb_idx], ids[prb_idx]
    sim = P @ G.T
    pos_list, neg_list = [], []
    # subsample probes for EER pair harvest (same as enrollment.py)
    eer_idx = np.arange(len(Pid))
    if len(eer_idx) > 3000:
        eer_idx = rng.choice(eer_idx, size=3000, replace=False)
    for i in eer_idx:
        same = Gid == Pid[i]
        if same.any():
            pos_list.append(sim[i, same])
        diff = ~same
        if diff.any():
            neg_scores = sim[i, diff]
            if len(neg_scores) > 64:
                neg_scores = rng.choice(neg_scores, size=64, replace=False)
            neg_list.append(neg_scores)
    pos = np.concatenate(pos_list) if pos_list else np.array([])
    neg = np.concatenate(neg_list) if neg_list else np.array([])
    e = eer(pos, neg)
    # top-1 retrieval on full sim
    top1_idx = np.argmax(sim, axis=1)
    top1 = float(np.mean(Gid[top1_idx] == Pid))
    return {
        "eer": round(e, 4),
        "top1": round(top1, 4),
        "n_ships": int(len(np.unique(Pid))),
        "n_probe": int(len(Pid)),
        "protocol": "clip_cosine",
        "disjointness_pass": True,
    }


def asnorm_set_eval(
    E: np.ndarray,
    ids: np.ndarray,
    k_enroll: int = K_ENROLL,
    probe_w: int = PROBE_W,
    seed: int = SEED,
) -> dict:
    """Same enroll/probe split as certified mean, scores via AS-norm (backend only)."""
    E = l2_normalize(E)
    ids = np.asarray(ids).astype(str)
    splits = _split_ships(ids, k_enroll, probe_w, seed)
    Pid = np.array(list(splits.keys()))
    P = np.stack([mean_proto(E, splits[s]["enroll"]) for s in Pid])
    pc = P @ P.T
    mu_e = pc.mean(1)
    sd_e = pc.std(1) + 1e-9
    pos, neg = [], []
    probes_v = []
    for s in Pid:
        for chunk in splits[s]["probe_chunks"]:
            v = mean_proto(E, chunk)
            probes_v.append((v, s))
            s_pe = P @ v
            mu_t = s_pe.mean()
            sd_t = s_pe.std() + 1e-9
            for k, pid in enumerate(Pid):
                z = 0.5 * ((s_pe[k] - mu_e[k]) / sd_e[k] + (s_pe[k] - mu_t) / sd_t)
                (pos if pid == s else neg).append(float(z))
    e = eer(np.array(pos), np.array(neg))
    # top-1 on raw cosine (AS-norm is for threshold calibration)
    return {
        "eer": round(e, 4),
        "top1": round(top1_protos(P, Pid, probes_v), 4),
        "n_ships": len(Pid),
        "n_probe": len(probes_v),
        "k_enroll": k_enroll,
        "probe_w": probe_w,
        "protocol": "certified_mean_asnorm",
        "disjointness_pass": True,
    }


def attention_set_eval(
    E: np.ndarray,
    ids: np.ndarray,
    attn_mod,
    device: str = "cpu",
    k_enroll: int = K_ENROLL,
    probe_w: int = PROBE_W,
    seed: int = SEED,
) -> dict:
    """Same split as certified; pool with gated attention instead of mean."""
    import torch

    E = l2_normalize(E)
    ids = np.asarray(ids).astype(str)
    splits = _split_ships(ids, k_enroll, probe_w, seed)
    attn_mod.eval()

    def pool(idx):
        with torch.no_grad():
            H = torch.from_numpy(E[idx].astype(np.float32)).unsqueeze(0).to(device)
            pooled, _ = attn_mod(H)
            return pooled.cpu().numpy()[0]

    Pid = np.array(list(splits.keys()))
    P = np.stack([pool(splits[s]["enroll"]) for s in Pid])
    probes = []
    for s in Pid:
        for chunk in splits[s]["probe_chunks"]:
            probes.append((pool(chunk), s))
    e, n = score_protos(P, Pid, probes)
    return {
        "eer": round(e, 4),
        "top1": round(top1_protos(P, Pid, probes), 4),
        "n_ships": len(Pid),
        "n_probe": n,
        "k_enroll": k_enroll,
        "probe_w": probe_w,
        "protocol": "attention_set_disjoint",
        "disjointness_pass": True,
    }


def stats_proto(E: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean+std (population) concat, then L2-normalize. Output dim = 2 * emb_dim."""
    H = E[idx]
    mu = H.mean(0)
    sigma = np.maximum(H.std(0, ddof=0), 1e-9)
    v = np.concatenate([mu, sigma])
    return v / (np.linalg.norm(v) + 1e-9)


def stats_set_eval(
    E: np.ndarray,
    ids: np.ndarray,
    k_enroll: int = K_ENROLL,
    probe_w: int = PROBE_W,
    seed: int = SEED,
) -> dict:
    """Mean+std statistics pooling for enroll proto AND each probe chunk.

    Same split as certified_mean_eval (_split_ships). Parameter-free.
    Pooled vectors are 2*emb_dim.
    """
    E = l2_normalize(E)
    ids = np.asarray(ids).astype(str)
    splits = _split_ships(ids, k_enroll, probe_w, seed)
    if len(splits) < 2:
        raise RuntimeError(f"too few usable ships: {len(splits)}")

    Pid = np.array(list(splits.keys()))
    P = np.stack([stats_proto(E, splits[s]["enroll"]) for s in Pid])
    probes = []
    for s in Pid:
        for chunk in splits[s]["probe_chunks"]:
            probes.append((stats_proto(E, chunk), s))

    e, n = score_protos(P, Pid, probes)
    return {
        "eer": round(e, 4),
        "top1": round(top1_protos(P, Pid, probes), 4),
        "n_ships": len(Pid),
        "n_probe": n,
        "k_enroll": k_enroll,
        "probe_w": probe_w,
        "dim": int(P.shape[1]),
        "disjointness_pass": True,
        "protocol": "certified_stats_pool",
    }


def asp_set_eval(
    E: np.ndarray,
    ids: np.ndarray,
    asp_mod,
    device: str = "cpu",
    k_enroll: int = K_ENROLL,
    probe_w: int = PROBE_W,
    seed: int = SEED,
) -> dict:
    """Same split as certified; pool with AttentiveStatsPool instead of mean."""
    import torch

    E = l2_normalize(E)
    ids = np.asarray(ids).astype(str)
    splits = _split_ships(ids, k_enroll, probe_w, seed)
    if len(splits) < 2:
        raise RuntimeError(f"too few usable ships: {len(splits)}")
    asp_mod.eval()

    def pool(idx):
        with torch.no_grad():
            H = torch.from_numpy(E[idx].astype(np.float32)).unsqueeze(0).to(device)
            pooled = asp_mod(H)
            if isinstance(pooled, tuple):
                pooled = pooled[0]
            return pooled.cpu().numpy()[0]

    Pid = np.array(list(splits.keys()))
    P = np.stack([pool(splits[s]["enroll"]) for s in Pid])
    probes = []
    for s in Pid:
        for chunk in splits[s]["probe_chunks"]:
            probes.append((pool(chunk), s))
    e, n = score_protos(P, Pid, probes)
    return {
        "eer": round(e, 4),
        "top1": round(top1_protos(P, Pid, probes), 4),
        "n_ships": len(Pid),
        "n_probe": n,
        "k_enroll": k_enroll,
        "probe_w": probe_w,
        "dim": int(P.shape[1]),
        "disjointness_pass": True,
        "protocol": "certified_asp",
    }


def aggregator_set_eval(
    E: np.ndarray,
    ids: np.ndarray,
    agg_mod,
    device: str = "cpu",
    k_enroll: int = K_ENROLL,
    probe_w: int = PROBE_W,
    seed: int = SEED,
    sessions=None,
) -> dict:
    """Same split as certified; pool enroll + probe chunks with a temporal aggregator.

    Clip-disjointness is asserted via _split_ships (raises on fail).
    If sessions is provided (aligned with E), also report soft session-disjoint
    stats: session_disjoint_pass is True only if for every ship,
    set(sessions[enroll]) ∩ set(sessions[probe_idxs]) == ∅.
    """
    import torch

    E = l2_normalize(E)
    ids = np.asarray(ids).astype(str)
    splits = _split_ships(ids, k_enroll, probe_w, seed)
    if len(splits) < 2:
        raise RuntimeError(f"too few usable ships: {len(splits)}")
    agg_mod.eval()

    def pool(idx):
        with torch.no_grad():
            H = torch.from_numpy(E[idx].astype(np.float32)).unsqueeze(0).to(device)
            pooled = agg_mod(H)
            if isinstance(pooled, tuple):
                pooled = pooled[0]
            return pooled.cpu().numpy()[0]

    Pid = np.array(list(splits.keys()))
    P = np.stack([pool(splits[s]["enroll"]) for s in Pid])
    probes = []
    for s in Pid:
        for chunk in splits[s]["probe_chunks"]:
            probes.append((pool(chunk), s))
    e, n = score_protos(P, Pid, probes)

    out = {
        "eer": round(e, 4),
        "top1": round(top1_protos(P, Pid, probes), 4),
        "n_ships": len(Pid),
        "n_probe": n,
        "k_enroll": k_enroll,
        "probe_w": probe_w,
        "dim": int(P.shape[1]),
        "disjointness_pass": True,
        "protocol": "certified_temporal_agg",
    }

    if sessions is not None:
        sess = np.asarray(sessions).astype(str)
        if len(sess) != len(ids):
            raise RuntimeError(
                f"sessions length {len(sess)} != embeddings/ids length {len(ids)}"
            )
        overlap_ships = 0
        n_probe_chunks = 0
        n_probe_chunks_overlap = 0
        for s in Pid:
            enroll = splits[s]["enroll"]
            enroll_sess = set(sess[enroll].tolist())
            ship_overlap = False
            for chunk in splits[s]["probe_chunks"]:
                n_probe_chunks += 1
                chunk_sess = set(sess[np.asarray(chunk)].tolist())
                if enroll_sess & chunk_sess:
                    n_probe_chunks_overlap += 1
                    ship_overlap = True
            if ship_overlap:
                overlap_ships += 1
        out["session_disjoint_pass"] = overlap_ships == 0
        out["session_overlap_ships"] = int(overlap_ships)
        out["session_overlap_probe_frac"] = (
            float(n_probe_chunks_overlap / n_probe_chunks)
            if n_probe_chunks > 0
            else 0.0
        )
    return out


def _pack_set_protocol(E, ids, pool_fn, k_enroll=K_ENROLL, probe_w=PROBE_W, seed=SEED):
    """Build Pid, prototypes, probe matrix, and per-probe ship indices."""
    E = l2_normalize(E)
    ids = np.asarray(ids).astype(str)
    splits = _split_ships(ids, k_enroll, probe_w, seed)
    Pid = np.array(list(splits.keys()))
    P = np.stack([pool_fn(E, splits[s]["enroll"]) for s in Pid])
    ship_to_i = {s: i for i, s in enumerate(Pid)}
    V, yi = [], []
    for s in Pid:
        for chunk in splits[s]["probe_chunks"]:
            V.append(pool_fn(E, chunk))
            yi.append(ship_to_i[s])
    return Pid, P, np.stack(V), np.asarray(yi, dtype=int)


def mmsi_bootstrap_ci(
    P: np.ndarray,
    yi: np.ndarray,
    sim: np.ndarray | None = None,
    V: np.ndarray | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """95% percentile CI by resampling evaluation MMSIs (vessels), not clips.

    Each bootstrap draws n_ships vessels with replacement, keeps the unique set,
    and recomputes EER / top-1 on gallery+probes restricted to that set.
    """
    n_ships = P.shape[0]
    if sim is None:
        if V is None:
            raise ValueError("need sim or V")
        sim = V @ P.T
    rng = np.random.default_rng(seed)
    eers, top1s, n_uniq = [], [], []

    def eer_fast(pos, neg, brng):
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        if len(neg) > 50 * len(pos):
            neg = brng.choice(neg, size=50 * len(pos), replace=False)
        s = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
        thr = np.quantile(s, np.linspace(0.0, 1.0, 400))
        best = (1.0, 0.5)
        for t in thr:
            pred = s >= t
            far = float(pred[y == 0].mean())
            frr = float((~pred[y == 1]).mean())
            d = abs(far - frr)
            if d < best[0]:
                best = (d, (far + frr) / 2)
        return float(best[1])

    for b in range(n_boot):
        drawn = rng.choice(n_ships, size=n_ships, replace=True)
        keep = np.unique(drawn)
        if keep.size < 2:
            continue
        mask = np.isin(yi, keep)
        if not mask.any():
            continue
        sim_b = sim[mask][:, keep]
        yi_b = yi[mask]
        col = {int(s): j for j, s in enumerate(keep)}
        yi_local = np.asarray([col[int(s)] for s in yi_b], dtype=int)
        pos = sim_b[np.arange(len(yi_local)), yi_local]
        neg_parts = []
        for j in range(keep.size):
            rows = yi_local == j
            if not rows.any():
                continue
            cols = [c for c in range(keep.size) if c != j]
            neg_parts.append(sim_b[rows][:, cols].ravel())
        neg = np.concatenate(neg_parts) if neg_parts else np.array([])
        brng = np.random.default_rng(seed + b + 1)
        eers.append(eer_fast(pos, neg, brng))
        pred = np.argmax(sim_b, axis=1)
        top1s.append(float(np.mean(pred == yi_local)))
        n_uniq.append(int(keep.size))
    ea = np.asarray(eers, float)
    ta = np.asarray(top1s, float)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "n_boot": int(len(ea)),
        "n_boot_requested": n_boot,
        "alpha": alpha,
        "eer_ci95": [round(float(np.percentile(ea, lo)), 4), round(float(np.percentile(ea, hi)), 4)],
        "top1_ci95": [round(float(np.percentile(ta, lo)), 4), round(float(np.percentile(ta, hi)), 4)],
        "eer_boot_mean": round(float(ea.mean()), 4),
        "top1_boot_mean": round(float(ta.mean()), 4),
        "mean_unique_ships": round(float(np.mean(n_uniq)), 2),
        "unit": "MMSI (vessel), resampled with replacement then unique",
    }


def hierarchical_mmsi_bootstrap(
    packs: list[dict],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Bootstrap: draw a split-seed pack, then resample MMSIs within that pack."""
    # Reuse mmsi path by concatenating with seed choice outside.
    rng = np.random.default_rng(seed)
    eers, top1s = [], []

    def eer_fast(pos, neg, brng):
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        if len(neg) > 50 * len(pos):
            neg = brng.choice(neg, size=50 * len(pos), replace=False)
        s = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
        thr = np.quantile(s, np.linspace(0.0, 1.0, 400))
        best = (1.0, 0.5)
        for t in thr:
            pred = s >= t
            far = float(pred[y == 0].mean())
            frr = float((~pred[y == 1]).mean())
            d = abs(far - frr)
            if d < best[0]:
                best = (d, (far + frr) / 2)
        return float(best[1])

    for b in range(n_boot):
        pack = packs[int(rng.integers(0, len(packs)))]
        P, yi, sim = pack["P"], pack["yi"], pack["sim"]
        n_ships = P.shape[0]
        drawn = rng.choice(n_ships, size=n_ships, replace=True)
        keep = np.unique(drawn)
        if keep.size < 2:
            continue
        mask = np.isin(yi, keep)
        if not mask.any():
            continue
        sim_b = sim[mask][:, keep]
        yi_b = yi[mask]
        col = {int(s): j for j, s in enumerate(keep)}
        yi_local = np.asarray([col[int(s)] for s in yi_b], dtype=int)
        pos = sim_b[np.arange(len(yi_local)), yi_local]
        neg_parts = []
        for j in range(keep.size):
            rows = yi_local == j
            if not rows.any():
                continue
            cols = [c for c in range(keep.size) if c != j]
            neg_parts.append(sim_b[rows][:, cols].ravel())
        neg = np.concatenate(neg_parts) if neg_parts else np.array([])
        eers.append(eer_fast(pos, neg, np.random.default_rng(seed + b + 17)))
        top1s.append(float(np.mean(np.argmax(sim_b, axis=1) == yi_local)))
    ea = np.asarray(eers, float)
    ta = np.asarray(top1s, float)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "n_boot": int(len(ea)),
        "n_boot_requested": n_boot,
        "alpha": alpha,
        "eer_ci95": [round(float(np.percentile(ea, lo)), 4), round(float(np.percentile(ea, hi)), 4)],
        "top1_ci95": [round(float(np.percentile(ta, lo)), 4), round(float(np.percentile(ta, hi)), 4)],
        "eer_boot_mean": round(float(ea.mean()), 4),
        "top1_boot_mean": round(float(ta.mean()), 4),
        "unit": "hierarchical: draw eval-seed pack, then resample MMSIs",
        "n_packs": len(packs),
    }
