#!/usr/bin/env python
"""
ShipNN reproduction + leakage comparison (VTUAD).

Design (confound-free): freeze ONE 100-ship population (subset-seed), then vary
ONLY the split rule and --seed (split assignment + torch init).
  --protocol random   : ShipNN-style clip-level 80/10/10 (leaky by construction)
  --protocol session  : per-ship session-disjoint 80/10/10 (same 100 classes,
                        different sub_init sessions in train vs val vs test)
  --protocol scenario : same 100 ships, train 2-4 km / val 3-5 km / test 4-6 km
                        (closed-set only on ships present in train AND test bands)

ALWAYS run `--dry-run` first: it builds the subset + both splits, audits class
coverage and session availability, writes manifests, and TRAINS NOTHING. If too
few of the 100 ships have >=2 sessions, the session experiment is underpowered
and you learn that in seconds instead of after 500 epochs.

>>> EDIT THIS ONE BLOCK to match your locked manifest column names <<<
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

# ---- repo paths ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.frontend import extract, in_channels          # noqa: E402
from src.models.shipnn import build_shipnn                      # noqa: E402

# ============================================================================
# CONFIG  ---  confirm/adjust these to YOUR repo, then nothing else needs edits
# ============================================================================
MANIFEST = ROOT / "data" / "metadata" / "vtuad_manifest.csv"    # master clip manifest
CACHE    = ROOT / "data" / "processed" / "feat_cache"           # .npy feature cache
REPORTS  = ROOT / "reports"

COLS = {            # logical name -> actual column in YOUR locked vtuad_manifest.csv
    "path":    "file_path",       # absolute path to the 1s wav
    "mmsi":    "mmsi",            # ship id; 0 == background (excluded from classes)
    "session": "session_id",      # recording-session id (= VTUAD sub_init)
    "date":    "recording_date",  # optional, for a later temporal protocol
    "scenario":"distance",        # distance band string (2-4 / 3-5 / 4-6 km)
}
N_SHIPS, N_SAMPLES = 100, 156
SUBSET_SEED = 1337            # frozen population — do not change between protocols
PLANNED_SEEDS = (1337, 2026, 42)  # preplanned split+init seeds for variance
# Cross-scenario distance bands (VTUAD inclusion radii)
DIST_TRAIN, DIST_VAL, DIST_TEST = "2-4 km", "3-5 km", "4-6 km"
SR_FALLBACK = 32000
# ============================================================================


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def set_run_seed(seed: int) -> None:
    """Split RNGs are separate; this only pins torch/numpy/python for training."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}\n"
                 f"-> set MANIFEST / COLS at the top of this script to your files.")
    df = pd.read_csv(MANIFEST)
    missing = [c for c in COLS.values() if c not in df.columns and c in
               (COLS["path"], COLS["mmsi"], COLS["session"])]
    if missing:
        sys.exit(f"manifest is missing required column(s) {missing}. "
                 f"Actual columns: {list(df.columns)}\n-> fix COLS at top.")
    df = df.rename(columns={v: k for k, v in COLS.items()})
    # Normalize types for grouping
    df["mmsi"] = df["mmsi"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["session"] = df["session"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "is_background" in df.columns:
        df = df[~df["is_background"].fillna(False).astype(bool)].copy()
    return df


def choose_ships(df: pd.DataFrame) -> list[str]:
    """Frozen ship list: MMSI≠0, ≥156 clips, top-100 by count (ties: mmsi asc)."""
    df = df[df["mmsi"].astype(str) != "0"].copy()
    counts = df.groupby("mmsi").size()
    eligible = counts[counts >= N_SAMPLES]
    if len(eligible) < N_SHIPS:
        sys.exit(f"only {len(eligible)} ships have >={N_SAMPLES} clips; "
                 f"cannot form the {N_SHIPS}-ship subset. Report this, do not fudge.")
    order = eligible.sort_values(ascending=False).rename("n_clips").reset_index()
    order = order.sort_values(["n_clips", "mmsi"], ascending=[False, True])
    return list(order["mmsi"].astype(str).head(N_SHIPS))


def build_subset(df: pd.DataFrame, subset_seed: int = SUBSET_SEED) -> pd.DataFrame:
    """Seeded 156-clip sample per chosen ship. Population is frozen by subset_seed."""
    df = df[df["mmsi"].astype(str) != "0"].copy()
    df["mmsi"] = df["mmsi"].astype(str)
    chosen = choose_ships(df)
    rng = np.random.default_rng(subset_seed)
    parts = [df[df["mmsi"] == m].sample(N_SAMPLES, random_state=int(rng.integers(1e9)))
             for m in chosen]
    sub = pd.concat(parts, ignore_index=True)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "shipnn_subset_mmsis.json").write_text(json.dumps(
        {"n_ships": N_SHIPS, "n_samples": N_SAMPLES, "subset_seed": subset_seed,
         "rule": "mmsi!=0; >=156 clips; top-100 by count (ties mmsi asc); seeded 156/ship",
         "mmsis": chosen}, indent=2))
    return sub


def split_random(sub: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for _, g in sub.groupby("mmsi"):            # stratified by ship, clip-level (leaky)
        idx = rng.permutation(len(g)); g = g.iloc[idx]
        n = len(g); a, b = int(.8 * n), int(.9 * n)
        out["train"].append(g.iloc[:a]); out["val"].append(g.iloc[a:b]); out["test"].append(g.iloc[b:])
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def split_session(sub: pd.DataFrame, seed: int):
    """Per-ship session-disjoint. Guarantees every class stays in train.

    VTUAD `session_id`/`sub_init` is NOT globally unique across ships, so
    grouping and leakage checks are always within a single MMSI.
    """
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    solo = 0
    for _, g in sub.groupby("mmsi"):
        sess = list(g["session"].unique())
        rng.shuffle(sess)
        if len(sess) == 1:                      # cannot be session-disjoint -> train only
            out["train"].append(g)
            solo += 1
            continue
        # assign whole sessions to reach ~80/10/10 of this ship's clips
        target = {"train": 0.8, "val": 0.1, "test": 0.1}
        clip_by_sess = g.groupby("session").size().to_dict()
        assign = {s: None for s in sess}
        assign[sess[0]] = "train"               # ensure class present in train
        rem = sess[1:]
        if len(sess) >= 3:
            assign[rem[-1]] = "test"
            assign[rem[-2]] = "val"
            rem = rem[:-2]
        elif len(sess) == 2:
            assign[rem[0]] = "test"
            rem = []
        cum = {"train": 0, "val": 0, "test": 0}
        for s, k in assign.items():
            if k is not None:
                cum[k] += clip_by_sess[s]
        for s in rem:
            k = min(
                target,
                key=lambda kk: (cum[kk] + clip_by_sess[s]) / max(target[kk], 1e-6),
            )
            assign[s] = k
            cum[k] += clip_by_sess[s]
        for s, k in assign.items():
            out[k].append(g[g["session"] == s])
    res = {
        k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=sub.columns))
        for k, v in out.items()
    }
    res["_solo_session_ships"] = solo
    return res


def split_scenario(pop: pd.DataFrame):
    """Distance-band split on the frozen 100 ships (all clips, not the 156 cap).

    Closed-set identity only for ships with ≥1 clip in train band AND test band.
    Val = mid band (may be empty for some ships). This is a domain-shift test,
    not a matched-N comparison to the 15,600-clip random/session cells.
    """
    if "scenario" not in pop.columns:
        sys.exit("manifest has no scenario/distance column after COLS rename.")
    pop = pop.copy()
    pop["scenario"] = pop["scenario"].astype(str)
    testable = (
        set(pop.loc[pop["scenario"] == DIST_TRAIN, "mmsi"])
        & set(pop.loc[pop["scenario"] == DIST_TEST, "mmsi"])
    )
    keep = pop[pop["mmsi"].isin(testable)]
    mapping = {DIST_TRAIN: "train", DIST_VAL: "val", DIST_TEST: "test"}
    parts = {}
    for dist, split in mapping.items():
        parts[split] = keep[keep["scenario"] == dist].reset_index(drop=True)
    parts["_n_frozen_ships"] = int(pop["mmsi"].nunique())
    parts["_n_testable"] = len(testable)
    parts["_n_sessions_multi_band"] = int(
        (keep.groupby(["mmsi", "session"])["scenario"].nunique() > 1).sum()
    )
    near_d = set(keep.loc[keep["scenario"] == DIST_TRAIN, "date"].astype(str))
    far_d = set(keep.loc[keep["scenario"] == DIST_TEST, "date"].astype(str))
    parts["_n_date_overlap_near_far"] = len(near_d & far_d)
    parts["_n_near_dates"] = len(near_d)
    parts["_n_far_dates"] = len(far_d)
    return parts


def audit(name, sp, sub):
    tr, va, te = sp["train"], sp["val"], sp["test"]
    classes = set(sub["mmsi"])
    rows = [
        f"### {name} split",
        f"- clips: train {len(tr)}, val {len(va)}, test {len(te)}",
        f"- classes in train/val/test: "
        f"{tr['mmsi'].nunique()}/{va['mmsi'].nunique()}/{te['mmsi'].nunique()} (of {len(classes)})",
    ]
    miss = classes - set(tr["mmsi"])
    if miss:
        rows.append(f"- !! {len(miss)} classes ABSENT from train (task ill-posed for these)")
    if name == "session":
        # Leakage must be checked WITHIN each ship (session ids collide across ships)
        leak = 0
        for m in classes:
            s_tr = set(tr.loc[tr["mmsi"] == m, "session"])
            s_hold = set(va.loc[va["mmsi"] == m, "session"]) | set(
                te.loc[te["mmsi"] == m, "session"]
            )
            leak += len(s_tr & s_hold)
        rows.append(f"- within-ship session leakage train~val/test: {leak}  (MUST be 0)")
        rows.append(f"- ships with only 1 session (train-only): {sp['_solo_session_ships']}")
        te_classes = te["mmsi"].nunique()
        rows.append(
            f"- ships testable in a different session: {te_classes}/{len(classes)}"
            + (
                "  <-- session experiment is well-powered"
                if te_classes >= 80
                else "  <-- WARNING: underpowered, few ships have multiple sessions"
            )
        )
        ratio = len(te) / max(len(tr) + len(va) + len(te), 1)
        rows.append(f"- test clip fraction: {ratio:.3f} (target ~0.10)")
    if name == "scenario":
        rows.append(
            f"- frozen 100-ship list; closed-set testable (near AND far): "
            f"{sp['_n_testable']}/{sp['_n_frozen_ships']}"
        )
        rows.append(
            f"- (mmsi, session) spanning >1 distance band: {sp['_n_sessions_multi_band']}  "
            f"<-- scenario is NOT a clean session-level domain cut"
        )
        rows.append(
            f"- recording-date overlap near~far: {sp['_n_date_overlap_near_far']} "
            f"(near {sp['_n_near_dates']} dates, far {sp['_n_far_dates']})"
        )
        rows.append(
            f"- train={DIST_TRAIN}  val={DIST_VAL}  test={DIST_TEST}  "
            "(do not compare N to the 15,600-clip random/session cells)"
        )
        if sp["_n_testable"] < 40:
            rows.append(
                "- WARNING: <40 ships in both bands — underpowered vs 100-class session cell"
            )
    return "\n".join(rows)


# ---- training (only runs without --dry-run) --------------------------------
# Module-level Dataset: Windows spawn workers cannot pickle nested classes/closures.
class ShipNNClipDataset:
    def __init__(self, df, cls_map, feature, train, aug, cache_dir: Path):
        self.paths = df["path"].astype(str).tolist()
        self.labels = [int(cls_map[m]) for m in df["mmsi"]]
        self.feature = feature
        self.train = bool(train)
        self.aug = bool(aug)
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.paths)

    def _feat(self, path: str):
        import hashlib
        import soundfile as sf
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
        key = self.cache_dir / f"{self.feature}_{digest}.npy"
        if key.exists():
            return np.load(key)
        y, sr = sf.read(path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        x = extract(y, sr if sr else SR_FALLBACK, self.feature)
        np.save(key, x)
        return x

    def __getitem__(self, i):
        import torch
        x = self._feat(self.paths[i]).copy()
        if self.train and self.aug:  # SpecAugment: mask freq + time
            rng = np.random.default_rng()
            f0 = rng.integers(0, max(1, x.shape[1] - 12))
            x[:, f0:f0 + 8, :] = 0
            t0 = rng.integers(0, max(1, x.shape[2] - 12))
            x[:, :, t0:t0 + 8] = 0
        return torch.from_numpy(x), self.labels[i]


def run_training(sp, sub, feature, epochs, lr, bs, aug, device, seed: int):
    import torch
    from torch.utils.data import DataLoader

    set_run_seed(seed)
    CACHE.mkdir(parents=True, exist_ok=True)
    mmsis = sorted(sub["mmsi"].unique())
    cls = {m: i for i, m in enumerate(mmsis)}
    g = torch.Generator()
    g.manual_seed(seed)

    def loader(df, train):
        # Windows: keep workers modest; 0 is safest if spawn still flakes
        nw = 0 if os.name == "nt" else 4
        return DataLoader(
            ShipNNClipDataset(df, cls, feature, train, aug, CACHE),
            batch_size=bs, shuffle=train, num_workers=nw,
            pin_memory=(device == "cuda"), drop_last=train,
            generator=g if train else None,
        )

    model = build_shipnn(len(mmsis), in_channels(feature)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    tl, vl, el = loader(sp["train"], True), loader(sp["val"], False), loader(sp["test"], False)

    def evaluate(dl):
        model.eval(); ok = tot = 0
        with torch.no_grad():
            for x, y in dl:
                p = model(x.to(device)).argmax(1).cpu()
                ok += (p == y).sum().item(); tot += len(y)
        return ok / max(tot, 1)

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tl:
            opt.zero_grad(); loss = crit(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs:
            v = evaluate(vl); best = max(best, v)
            log(f"epoch {ep}/{epochs}  val_acc={v:.4f}  best={best:.4f}")
    return evaluate(el), best


def write_seed_variance_report(df: pd.DataFrame) -> None:
    """Honest n-seed table for random vs session. Scenario is a different experiment."""
    rs = df[(df["protocol"].isin(["random", "session"])) & (df["feature"] == "cqt_mfcc")]
    if rs.empty:
        return
    lines = [
        "# Seed-variance — random vs session (closed-set)",
        "",
        "Frozen 100-ship × 156-clip population (`subset_seed=1337`). "
        "`seed` controls split assignment and torch init only.",
        "A null result is stated with spread, not a single 0.9-point delta.",
        "",
    ]
    try:
        lines.append(rs.sort_values(["protocol", "seed"]).to_markdown(index=False))
    except ImportError:
        lines.append(rs.sort_values(["protocol", "seed"]).to_string(index=False))
    lines.append("")
    for proto in ("random", "session"):
        g = rs[rs["protocol"] == proto]["test_acc"]
        if len(g) == 0:
            continue
        std = g.std(ddof=1) if len(g) > 1 else float("nan")
        lines.append(
            f"- **{proto}**: n={len(g)}  mean={g.mean():.4f}  "
            f"std={std:.4f}  min={g.min():.4f}  max={g.max():.4f}"
        )
    merged = rs[rs.protocol == "random"][["seed", "test_acc"]].merge(
        rs[rs.protocol == "session"][["seed", "test_acc"]],
        on="seed", suffixes=("_random", "_session"),
    )
    if len(merged):
        delta = merged["test_acc_random"] - merged["test_acc_session"]
        lines.append(
            "- **Δ (random − session) by seed**: "
            + ", ".join(f"{int(se)}:{d:+.4f}" for se, d in zip(merged["seed"], delta))
        )
        if len(delta) > 1:
            lines.append(
                f"- **Δ mean ± sample std**: {delta.mean():+.4f} ± {delta.std(ddof=1):.4f}"
            )
        else:
            lines.append(f"- **Δ (n=1)**: {delta.mean():+.4f}  — not a variance estimate")
    (REPORTS / "seed_variance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result(protocol, feature, aug, test_acc, val_best, seed: int,
                 subset_seed: int = SUBSET_SEED):
    REPORTS.mkdir(exist_ok=True)
    csvp = REPORTS / "reproduction_results.csv"
    row = dict(protocol=protocol, feature=feature, aug=aug, seed=seed,
               subset_seed=subset_seed,
               test_acc=round(test_acc, 4), val_best=round(val_best, 4),
               ts=time.strftime("%Y-%m-%d %H:%M"))
    df = pd.read_csv(csvp) if csvp.exists() else pd.DataFrame()
    if not df.empty:
        if "seed" not in df.columns:
            df["seed"] = SUBSET_SEED
        if "subset_seed" not in df.columns:
            df["subset_seed"] = SUBSET_SEED
        mask = pd.Series(True, index=df.index)
        for k in ("protocol", "feature", "aug", "seed"):
            mask &= df[k].astype(str) == str(row[k])
        df = df.loc[~mask]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csvp, index=False)
    write_seed_variance_report(df)
    last = df.sort_values("ts").groupby(
        ["feature", "aug", "protocol"], as_index=False
    ).tail(1)
    piv = last.pivot_table(index=["feature", "aug"], columns="protocol",
                           values="test_acc", aggfunc="last")
    try:
        table = piv.to_markdown()
    except ImportError:
        table = piv.to_string()
    n_seed = df[df["protocol"].isin(["random", "session"])]["seed"].nunique()
    (REPORTS / "reproduction.md").write_text(
        "# ShipNN reproduction — test accuracy\n\n"
        "Same 100-ship x 156-clip population; only the split rule changes.\n"
        "`random` is the leaky ShipNN-style baseline; `session` is leakage-free "
        "closed-set ID. MMSI-disjoint is NOT here — it is a verification task, "
        "not closed-set accuracy.\n\n"
        f"Latest cell per protocol (seed-variance: `{n_seed}` seeds; "
        "see `seed_variance.md`).\n\n" + table + "\n",
        encoding="utf-8",
    )
    log(f"wrote {csvp} and {REPORTS/'reproduction.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["random", "session", "scenario"], default="random")
    ap.add_argument("--feature", choices=["cqt", "mfcc", "cqt_mfcc"], default="cqt_mfcc")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--seed", type=int, default=SUBSET_SEED,
                    help="split assignment + torch init (not the ship population)")
    ap.add_argument("--subset-seed", type=int, default=SUBSET_SEED,
                    help="frozen 156-clip sample; keep 1337 across protocols")
    ap.add_argument("--dry-run", action="store_true", help="build+audit splits, no training")
    a = ap.parse_args()

    df = load_manifest()
    log(f"manifest: {len(df)} clips, {df['mmsi'].nunique()} raw mmsi values")
    ships = choose_ships(df[df["mmsi"].astype(str) != "0"])
    clip_sub = build_subset(df, a.subset_seed)
    log(f"subset: {len(clip_sub)} clips, {clip_sub['mmsi'].nunique()} ships "
        f"(expected {N_SHIPS}x{N_SAMPLES}={N_SHIPS * N_SAMPLES}); "
        f"subset_seed={a.subset_seed} run_seed={a.seed}")

    if a.protocol == "scenario":
        pop = df[df["mmsi"].isin(ships)].copy()
        splits = split_scenario(pop)
        eval_pop = pd.concat(
            [splits["train"], splits["val"], splits["test"]], ignore_index=True
        )
        log(f"scenario population: {len(pop)} clips of {len(ships)} ships; "
            f"closed-set testable={splits['_n_testable']}")
    else:
        eval_pop = clip_sub
        splits = {"random": split_random, "session": split_session}[a.protocol](
            clip_sub, a.seed
        )

    report = audit(a.protocol, splits, eval_pop)
    print("\n" + report + "\n")
    (REPORTS / f"split_audit_{a.protocol}_subset.md").write_text(report + "\n")
    out_dir = REPORTS / "seed_runs"
    out_dir.mkdir(exist_ok=True)
    for k in ("train", "val", "test"):
        splits[k].to_csv(out_dir / f"{a.protocol}_s{a.seed}_{k}.csv", index=False)
        if a.seed == SUBSET_SEED:
            splits[k].to_csv(REPORTS / f"{a.protocol}_{k}_subset.csv", index=False)

    if a.dry_run:
        log("DRY RUN complete — no model trained. Inspect the audit above first.")
        return

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        log("WARNING: no CUDA — training will be slow.")
    test_acc, val_best = run_training(
        splits, eval_pop, a.feature, a.epochs, a.lr, a.bs, a.aug, dev, a.seed
    )
    log(f"RESULT protocol={a.protocol} feature={a.feature} aug={a.aug} "
        f"seed={a.seed} test_acc={test_acc:.4f} val_best={val_best:.4f}")
    write_result(a.protocol, a.feature, a.aug, test_acc, val_best, a.seed, a.subset_seed)


if __name__ == "__main__":
    main()
