"""
Open-set enrollment/retrieval evaluation for unseen vessels (VTUAD MMSI-disjoint).

Honest protocols (VTUAD `sub_init` is often 1 clip — do NOT treat it as a session):

  clip_split   : per ship, random half gallery / half probe
                 -> primary retrieval + verification + OSCR vs background
  enroll_k     : per ship, k gallery clips, rest probe (enrollment-style)
  across_date  : gallery and probe on DIFFERENT recording dates
                 -> honest condition shift; tiny N (~3 ships on VTUAD test)

Metrics: EER, top-1 / top-5, OSCR. Never train here.
"""
from __future__ import annotations
import numpy as np


def _cosine(a, b):
    return a @ b.T


def eer(scores_pos, scores_neg):
    """Equal Error Rate from genuine vs impostor cosine scores."""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    # Cap impostors so mega-ships don't dominate the threshold search
    neg = scores_neg
    if len(neg) > 50 * len(scores_pos):
        rng = np.random.default_rng(0)
        neg = rng.choice(neg, size=50 * len(scores_pos), replace=False)
    s = np.concatenate([scores_pos, neg])
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(neg))])
    thr = np.unique(s)
    if len(thr) > 2000:
        thr = np.quantile(s, np.linspace(0, 1, 2000))
    best = (1.0, 0.5)
    for t in thr:
        pred = s >= t
        far = float(np.mean(pred[y == 0]))
        frr = float(np.mean(~pred[y == 1]))
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return float(best[1])


def topk_retrieval(sim, probe_ids, gallery_ids, ks=(1, 5)):
    """Vectorized top-k hit rate (argpartition — avoids full argsort on huge NxM)."""
    probe_ids = np.asarray(probe_ids)
    gallery_ids = np.asarray(gallery_ids)
    out = {}
    kmax = int(max(ks))
    # Take indices of kmax largest sims per row (unordered), then sort those only
    part = np.argpartition(-sim, kth=min(kmax, sim.shape[1] - 1), axis=1)[:, :kmax]
    row = np.arange(sim.shape[0])[:, None]
    top_sims = sim[row, part]
    order = np.argsort(-top_sims, axis=1)
    top_ids = gallery_ids[part[row, order]]
    for k in ks:
        hit = np.any(top_ids[:, :k] == probe_ids[:, None], axis=1)
        out[k] = float(np.mean(hit))
    return out


def oscr(sim_known, known_correct, sim_unknown):
    thr = np.unique(np.concatenate([sim_known, sim_unknown]))
    if len(thr) > 2000:
        thr = np.quantile(np.concatenate([sim_known, sim_unknown]), np.linspace(0, 1, 2000))
    pts = []
    for t in thr:
        acc_known = sim_known >= t
        ccr = float(np.mean(known_correct & acc_known)) if len(sim_known) else 0.0
        fpr = float(np.mean(sim_unknown >= t)) if len(sim_unknown) else 0.0
        pts.append((fpr, ccr))
    pts = sorted(pts)
    fpr_arr = np.array([p[0] for p in pts])
    ccr_arr = np.array([p[1] for p in pts])
    ccr_at_10 = float(ccr_arr[np.argmin(np.abs(fpr_arr - 0.1))])
    # Manual trapezoid (avoid numpy 1/2 trapz vs trapezoid rename issues)
    order = np.argsort(fpr_arr)
    fpr_s, ccr_s = fpr_arr[order], ccr_arr[order]
    auc = float(np.sum((fpr_s[1:] - fpr_s[:-1]) * (ccr_s[1:] + ccr_s[:-1]) / 2.0))
    return {"ccr@fpr0.1": round(ccr_at_10, 4), "oscr_auc": round(auc, 4)}


def _build_pairs(ids, dates, protocol, rng, enroll_k=5):
    """Return gallery/probe index arrays and list of used ship ids."""
    ids = np.asarray(ids).astype(str)
    dates = None if dates is None else np.asarray(dates).astype(str)
    gal, prb, used = [], [], []
    for ship in np.unique(ids):
        idx = np.where(ids == ship)[0]
        if protocol == "clip_split":
            if len(idx) < 2:
                continue
            rng.shuffle(idx)
            cut = max(1, len(idx) // 2)
            gal += list(idx[:cut])
            prb += list(idx[cut:])
            used.append(ship)
        elif protocol == "enroll_k":
            if len(idx) < enroll_k + 1:
                continue
            rng.shuffle(idx)
            gal += list(idx[:enroll_k])
            prb += list(idx[enroll_k:])
            used.append(ship)
        elif protocol == "across_date":
            if dates is None:
                continue
            d = dates[idx]
            uniq = np.unique(d)
            if len(uniq) < 2:
                continue
            rng.shuffle(uniq)
            g_dates = set(uniq[: max(1, len(uniq) // 2)])
            g = idx[np.isin(d, list(g_dates))]
            p = idx[~np.isin(d, list(g_dates))]
            if len(g) < 1 or len(p) < 1:
                continue
            gal += list(g)
            prb += list(p)
            used.append(ship)
        else:
            raise ValueError(f"unknown protocol {protocol}")
    return np.asarray(gal, int), np.asarray(prb, int), used


def evaluate(emb, ids, dates=None, bg_emb=None, protocol="clip_split",
             enroll_k=5, seed=0, max_probes_for_eer=3000, max_impostor_pairs=200_000):
    ids = np.asarray(ids).astype(str)
    dates = None if dates is None else np.asarray(dates).astype(str)
    rng = np.random.default_rng(seed)
    gi, pi, used = _build_pairs(ids, dates, protocol, rng, enroll_k=enroll_k)
    if len(used) < 2:
        return {
            "protocol": protocol,
            "usable_ships": len(used),
            "note": "too few usable ships",
        }

    G, Gid = emb[gi], ids[gi]
    P, Pid = emb[pi], ids[pi]
    sim = _cosine(P, G)

    # Subsample probes for EER pair harvesting (topk still uses full sim)
    eer_idx = np.arange(len(Pid))
    if len(eer_idx) > max_probes_for_eer:
        eer_idx = rng.choice(eer_idx, size=max_probes_for_eer, replace=False)
    pos_list, neg_list = [], []
    for i in eer_idx:
        pid = Pid[i]
        same = Gid == pid
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
    if len(neg) > max_impostor_pairs:
        neg = rng.choice(neg, size=max_impostor_pairs, replace=False)

    metrics = {
        "protocol": protocol,
        "usable_ships": len(used),
        "n_probe": int(len(pi)),
        "n_gallery": int(len(gi)),
        "eer": round(eer(pos, neg), 4),
    }
    metrics.update({f"top{k}": round(v, 4) for k, v in topk_retrieval(sim, Pid, Gid).items()})

    top1_idx = np.argmax(sim, axis=1)
    known_max = sim[np.arange(len(P)), top1_idx]
    known_correct = Gid[top1_idx] == Pid
    if bg_emb is not None and len(bg_emb):
        unk_max = _cosine(bg_emb, G).max(axis=1)
        metrics.update(oscr(known_max, known_correct, unk_max))
    return metrics


def report(emb, ids, sessions=None, dates=None, bg_emb=None, seed=0, enroll_k=5):
    """
    Primary: clip_split + enroll_k + OSCR.
    Secondary: across_date (honest when N allows).
    `sessions` is accepted for API compat but ignored (sub_init is not a session).
    """
    del sessions  # intentionally unused — VTUAD sub_init ≠ recording session
    clip = evaluate(emb, ids, dates, bg_emb, "clip_split", enroll_k, seed)
    enk = evaluate(emb, ids, dates, bg_emb, "enroll_k", enroll_k, seed)
    ad = evaluate(emb, ids, dates, bg_emb, "across_date", enroll_k, seed)

    # Compat aliases so old score scripts / notes still parse something
    out = {
        "clip_split": clip,
        "enroll_k": enk,
        "across_date": ad,
        # legacy keys (clip_split ≈ old within_session; across_date is the honest gap)
        "within_session": clip,
        "across_session": ad,
        "across_session_date_disjoint": ad,
        "note": (
            "sub_init-based across_session removed: VTUAD sub_init is often 1 clip. "
            "Primary metrics: clip_split / enroll_k (+ OSCR). "
            "Honest condition shift: across_date (tiny N)."
        ),
    }
    if "eer" in clip and "eer" in ad and not np.isnan(ad.get("eer", np.nan)):
        out["eer_gap_across_date_minus_clip"] = round(ad["eer"] - clip["eer"], 4)
        out["eer_gap_across_minus_within"] = out["eer_gap_across_date_minus_clip"]
    else:
        out["eer_gap_across_date_minus_clip"] = None
        out["eer_gap_across_minus_within"] = None
    return out
