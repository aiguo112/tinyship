#!/usr/bin/env python
"""
E2.1 — Train SpeechBrain ECAPA-TDNN on native fbank + AAM-softmax, embed test,
save ckpt+embeddings BEFORE certified scoring.

Protocol (locked):
  split: data/processed/splits/mmsi_{train,test}.csv (178 / 22 MMSI-disjoint)
  frontend: fbank 16 kHz, 80 mel, win 25 ms, hop 10 ms, per-utt mean subtract
  label: ECAPA-TDNN (fbank)
  objective: SpeechBrain AAM-softmax (AdditiveAngularMargin), NOT SubCenter
  eval: certified_mean_eval + stats_set_eval, k_enroll=20, probe_w=3, seeds 0-4
  assert n_ships==22, n_probe==7194; enroll∩probe clip-disjoint

  python scripts/reproduce/train_e2_ecapa.py --dry-run
  python scripts/reproduce/train_e2_ecapa.py --bs 128 --epochs 120
  python scripts/reproduce/train_e2_ecapa.py --score-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SPLIT_DIR = ROOT / "data" / "processed" / "splits"
CACHE = ROOT / "data" / "processed" / "feat_cache"
REPORTS = ROOT / "reports"
CKPT_PATH = REPORTS / "e2_ecapa_ckpt.pt"
EMB_PATH = REPORTS / "e2_ecapa_embeddings.npz"
JSON_PATH = REPORTS / "e2_ecapa.json"

COLS = {"path": "file_path", "mmsi": "MMSI", "session": "sub_init", "date": "date"}
CACHE_PREFIX = "fbank16k80_"

SR_FBANK = 16000
N_MELS = 80
WIN_MS = 25
HOP_MS = 10
N_FFT = 400
HOP_LENGTH = int(SR_FBANK * HOP_MS / 1000)  # 160
WIN_LENGTH = int(SR_FBANK * WIN_MS / 1000)  # 400
EMB_DIM = 192
# SpeechBrain VoxCeleb / ECAPA recipe AAM (class ctor defaults are 0/1; recipe uses these)
AAM_MARGIN = 0.2
AAM_SCALE = 30.0

K_ENROLL = 20
PROBE_W = 3
SEEDS = [0, 1, 2, 3, 4]
N_SHIPS_REF = 22
N_PROBE_REF = 7194
OURS_STATS_EER = 0.0629
OURS_PARAMS = 98640


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _shim_torchaudio() -> None:
    import torchaudio

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]


def load_split(name: str) -> pd.DataFrame:
    p = SPLIT_DIR / f"mmsi_{name}.csv"
    if not p.exists():
        sys.exit(f"missing split file {p}")
    df = pd.read_csv(p).rename(columns={v: k for k, v in COLS.items()})
    df = df[df["mmsi"].astype(str) != "0"].copy()
    df["mmsi"] = df["mmsi"].astype(str)
    return df


def load_background(name: str) -> pd.DataFrame:
    p = SPLIT_DIR / f"mmsi_{name}.csv"
    df = pd.read_csv(p).rename(columns={v: k for k, v in COLS.items()})
    return df[df["mmsi"].astype(str) == "0"].copy()


def _cache_key(path: str) -> Path:
    h = hashlib.sha256(path.encode()).hexdigest()[:16]
    return CACHE / f"{CACHE_PREFIX}{h}.npy"


def extract_fbank_numpy(path: str) -> np.ndarray:
    """Return (T, 80) float32 fbank with per-utterance mean subtract. Cached."""
    key = _cache_key(path)
    if key.exists():
        return np.load(key)

    import soundfile as sf
    import torch
    import torch.nn.functional as F
    import torchaudio

    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    wav = torch.from_numpy(y)
    if sr != SR_FBANK:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SR_FBANK).squeeze(0)
    n = SR_FBANK
    if wav.numel() > n:
        wav = wav[:n]
    elif wav.numel() < n:
        wav = F.pad(wav, (0, n - wav.numel()))

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR_FBANK,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=20.0,
        f_max=SR_FBANK / 2,
        power=2.0,
        center=True,
        norm="slaney",
        mel_scale="slaney",
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    with torch.no_grad():
        m = to_db(mel(wav.unsqueeze(0)))  # (1, n_mels, frames)
        m = m - m.mean(dim=-1, keepdim=True)
        feats = m.transpose(1, 2).squeeze(0).numpy().astype(np.float32)  # (T, F)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(key, feats)
    return feats


class EcapaFbankDataset:
    """Module-level for Windows spawn safety."""

    def __init__(self, df: pd.DataFrame, cmap: dict, labeled: bool):
        self.paths = df["path"].astype(str).tolist()
        self.mmsis = df["mmsi"].astype(str).tolist()
        self.cmap = dict(cmap)
        self.labeled = bool(labeled)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch

        x = extract_fbank_numpy(self.paths[i]).copy()
        y = int(self.cmap[self.mmsis[i]]) if self.labeled else -1
        return torch.from_numpy(x), y


def _lr_at(epoch: int, epochs: int, base_lr: float, warmup: int = 5) -> float:
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    t = (epoch - warmup) / max(epochs - warmup, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * t))


def nparams(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _dj(r: dict) -> str:
    return "PASS" if r.get("disjointness_pass") else "FAIL"


def _mean_std(xs):
    arr = np.array(xs, dtype=float)
    return round(float(arr.mean()), 4), round(float(arr.std(ddof=1)), 4)


def score_embeddings(E: np.ndarray, ids: np.ndarray) -> dict:
    from src.evaluation.certified_verify import certified_mean_eval, stats_set_eval

    ids = np.asarray(ids).astype(str)
    mean_eers, mean_top1s = [], []
    stats_eers, stats_top1s = [], []
    clip_status = "PASS"
    n_ships = n_probe = None

    for seed in SEEDS:
        rm = certified_mean_eval(E, ids, K_ENROLL, PROBE_W, seed)
        rs = stats_set_eval(E, ids, K_ENROLL, PROBE_W, seed)
        for name, r in (("mean_pool", rm), ("stats_pool", rs)):
            if r["n_ships"] != N_SHIPS_REF:
                sys.exit(
                    f"PROTOCOL MISMATCH: {name} seed={seed} n_ships={r['n_ships']} "
                    f"want {N_SHIPS_REF}"
                )
            if r["n_probe"] != N_PROBE_REF:
                sys.exit(
                    f"PROTOCOL MISMATCH: {name} seed={seed} n_probe={r['n_probe']} "
                    f"want {N_PROBE_REF}"
                )
            dj = _dj(r)
            log(
                f"  {name:10} seed={seed} n_ships={r['n_ships']} n_probe={r['n_probe']} "
                f"disjoint={dj} EER={r['eer']} top1={r['top1']}"
            )
            if dj != "PASS":
                clip_status = "FAIL"
                sys.exit(f"DISJOINTNESS FAIL {name} seed={seed}")
        mean_eers.append(rm["eer"])
        mean_top1s.append(rm["top1"])
        stats_eers.append(rs["eer"])
        stats_top1s.append(rs["top1"])
        n_ships = rm["n_ships"]
        n_probe = rm["n_probe"]

    mm_eer, mm_std = _mean_std(mean_eers)
    mm_t1, mm_t1_std = _mean_std(mean_top1s)
    ss_eer, ss_std = _mean_std(stats_eers)
    ss_t1, ss_t1_std = _mean_std(stats_top1s)
    ecapa_best = min(mm_eer, ss_eer)

    return {
        "protocol_label": "ECAPA-TDNN (fbank)",
        "objective": (
            f"SpeechBrain AAM-softmax via LogSoftmaxWrapper(AdditiveAngularMargin"
            f"(margin={AAM_MARGIN}, scale={AAM_SCALE})); NOT SubCenter ArcFace"
        ),
        "n_ships": int(n_ships),
        "n_probe": int(n_probe),
        "clip_disjoint": clip_status,
        "mean_pool": {
            "eer_mean": mm_eer,
            "eer_std": mm_std,
            "eers": mean_eers,
            "top1_mean": mm_t1,
            "top1_std": mm_t1_std,
            "top1s": mean_top1s,
        },
        "stats_pool": {
            "eer_mean": ss_eer,
            "eer_std": ss_std,
            "eers": stats_eers,
            "top1_mean": ss_t1,
            "top1_std": ss_t1_std,
            "top1s": stats_top1s,
        },
        "vs_ours_stats_pool": {
            "ours_eer": OURS_STATS_EER,
            "ecapa_best_eer": ecapa_best,
            "delta": round(float(ecapa_best - OURS_STATS_EER), 4),
            "params_ratio": round(6194048 / OURS_PARAMS, 1),
        },
    }


def merge_report(e2_1: dict) -> dict:
    base = {}
    if JSON_PATH.exists():
        base = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    base["experiment"] = "E2"
    base["status"] = "e2_1_complete"
    base["e2_1"] = e2_1
    base["ready_for_e3"] = e2_1.get("clip_disjoint") == "PASS" and e2_1.get(
        "mean_pool", {}
    ).get("eer_mean") is not None
    JSON_PATH.write_text(json.dumps(base, indent=2), encoding="utf-8")
    log(f"wrote {JSON_PATH}")
    return base


def run_score_only() -> dict:
    if not EMB_PATH.exists():
        sys.exit(f"missing embeddings {EMB_PATH}; train first")
    d = np.load(EMB_PATH, allow_pickle=True)
    E, ids = d["E"], d["ids"]
    log(f"loaded {EMB_PATH} E={E.shape}")
    e2_1 = score_embeddings(E, ids)

    # fill params / train meta from ckpt if present
    params_n = 6194048
    epochs = bs = wall = None
    if CKPT_PATH.exists():
        import torch

        ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        params_n = int(cfg.get("ecapa_params", params_n))
        epochs = cfg.get("epochs")
        bs = cfg.get("bs")
        wall = cfg.get("wall_clock_hours")
        if "aam_margin" in cfg:
            e2_1["objective"] = (
                f"SpeechBrain AAM-softmax via LogSoftmaxWrapper(AdditiveAngularMargin"
                f"(margin={cfg['aam_margin']}, scale={cfg['aam_scale']})); "
                f"NOT SubCenter ArcFace"
            )
        e2_1["vs_ours_stats_pool"]["params_ratio"] = round(params_n / OURS_PARAMS, 1)

    e2_1["params"] = params_n
    e2_1["epochs"] = epochs
    e2_1["bs"] = bs
    e2_1["wall_clock_hours"] = wall
    e2_1["commands"] = [
        r". .\scripts\use_tinyship.ps1",
        "python scripts/reproduce/train_e2_ecapa.py --bs 128 --epochs 120",
        "python scripts/reproduce/score_e2_ecapa.py",
    ]
    merge_report(e2_1)
    return e2_1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=AAM_MARGIN)
    ap.add_argument("--scale", type=float, default=AAM_SCALE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--skip-score", action="store_true",
                    help="train+embed+save only; do not score")
    a = ap.parse_args()

    if a.score_only:
        e2_1 = run_score_only()
        mp, sp = e2_1["mean_pool"], e2_1["stats_pool"]
        log(
            f"RESULT mean_pool EER={mp['eer_mean']}±{mp['eer_std']} "
            f"top1={mp['top1_mean']}±{mp['top1_std']}"
        )
        log(
            f"RESULT stats_pool EER={sp['eer_mean']}±{sp['eer_std']} "
            f"top1={sp['top1_mean']}±{sp['top1_std']}"
        )
        return

    tr = load_split("train")
    te = load_split("test")
    bg_te = load_background("test")
    n_train_ships = int(tr["mmsi"].nunique())
    n_test_ships = int(te["mmsi"].nunique())
    log(
        f"train ships {n_train_ships} clips {len(tr)} | "
        f"test ships {n_test_ships} clips {len(te)} | "
        f"test background {len(bg_te)}"
    )
    if n_train_ships != 178:
        log(f"WARNING: expected 178 train ships, got {n_train_ships}")
    if n_test_ships != 22:
        log(f"WARNING: expected 22 test ships, got {n_test_ships}")

    # dry-run: one fbank + model shapes
    sample_path = tr["path"].astype(str).iloc[0]
    feats0 = extract_fbank_numpy(sample_path)
    log(
        f"frontend=fbank sr={SR_FBANK} n_mels={N_MELS} "
        f"win={WIN_MS}ms hop={HOP_MS}ms frames={feats0.shape[0]} "
        f"shape={feats0.shape} cache_prefix={CACHE_PREFIX}"
    )

    if a.dry_run:
        _shim_torchaudio()
        import torch
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier
        from speechbrain.nnet.losses import AdditiveAngularMargin, LogSoftmaxWrapper

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ecapa = ECAPA_TDNN(input_size=N_MELS, lin_neurons=EMB_DIM).to(device)
        clf = Classifier(
            input_size=EMB_DIM, lin_neurons=EMB_DIM, out_neurons=n_train_ships
        ).to(device)
        aam = LogSoftmaxWrapper(
            AdditiveAngularMargin(margin=a.margin, scale=a.scale)
        ).to(device)
        # BN needs B>1 in train mode; dry-run uses eval + B=2
        ecapa.eval()
        clf.eval()
        x = torch.from_numpy(feats0).unsqueeze(0).repeat(2, 1, 1).to(device)
        lengths = torch.ones(2, device=device)
        with torch.no_grad():
            emb = ecapa(x, lengths=lengths)
            e_in = emb.unsqueeze(1) if emb.dim() == 2 else emb
            loss = aam(
                clf(e_in),
                torch.zeros(2, 1, dtype=torch.long, device=device),
            )
        log(
            f"DRY RUN ok emb={tuple(emb.shape)} loss={float(loss):.4f} "
            f"ecapa_params={nparams(ecapa)} clf_params={nparams(clf)} "
            f"aam margin={a.margin} scale={a.scale}"
        )
        return

    _shim_torchaudio()
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier
    from speechbrain.nnet.losses import AdditiveAngularMargin, LogSoftmaxWrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log("WARNING: no CUDA — training will be very slow")
    else:
        log(f"device=cuda name={torch.cuda.get_device_name(0)}")

    CACHE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    nw = 0 if os.name == "nt" else 4

    classes = sorted(tr["mmsi"].unique())
    cmap = {m: i for i, m in enumerate(classes)}
    labels = np.array([cmap[m] for m in tr["mmsi"]], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(classes)).astype(np.float64)
    w_per_class = 1.0 / np.maximum(counts, 1.0)
    sample_w = w_per_class[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )

    ecapa = ECAPA_TDNN(input_size=N_MELS, lin_neurons=EMB_DIM).to(device)
    clf = Classifier(
        input_size=EMB_DIM, lin_neurons=EMB_DIM, out_neurons=len(classes)
    ).to(device)
    aam = LogSoftmaxWrapper(
        AdditiveAngularMargin(margin=a.margin, scale=a.scale)
    ).to(device)
    ecapa_n = nparams(ecapa)
    log(
        f"ecapa_params={ecapa_n} clf_params={nparams(clf)} "
        f"AAM margin={a.margin} scale={a.scale} epochs={a.epochs} bs={a.bs} lr={a.lr}"
    )

    opt = torch.optim.Adam(
        list(ecapa.parameters()) + list(clf.parameters()), lr=a.lr
    )
    dl = DataLoader(
        EcapaFbankDataset(tr, cmap, True),
        batch_size=a.bs,
        sampler=sampler,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    t_train0 = time.perf_counter()
    try:
        for ep in range(1, a.epochs + 1):
            lr = _lr_at(ep, a.epochs, a.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            ecapa.train()
            clf.train()
            running = 0.0
            n_batches = 0
            for x, y in dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                lengths = torch.ones(x.size(0), device=device)
                emb = ecapa(x, lengths=lengths)
                e_in = emb.unsqueeze(1) if emb.dim() == 2 else emb
                loss = aam(clf(e_in), y.unsqueeze(1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += float(loss.item())
                n_batches += 1
            if ep % 10 == 0 or ep == a.epochs or ep == 1:
                log(
                    f"epoch {ep}/{a.epochs} loss {running / max(n_batches, 1):.4f} "
                    f"lr={lr:.2e}"
                )
    except torch.cuda.OutOfMemoryError:
        log(f"OOM at bs={a.bs}. Retry once with smaller batch (see caller).")
        raise

    wall_train = (time.perf_counter() - t_train0) / 3600.0
    log(f"train wall_clock_hours={wall_train:.3f}")

    @torch.no_grad()
    def embed(df: pd.DataFrame) -> np.ndarray:
        ecapa.eval()
        out = []
        loader = DataLoader(
            EcapaFbankDataset(df, cmap, False),
            batch_size=256,
            num_workers=nw,
            pin_memory=(device.type == "cuda"),
        )
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            lengths = torch.ones(x.size(0), device=device)
            emb = ecapa(x, lengths=lengths)
            if emb.dim() == 3:
                emb = emb.squeeze(1)
            out.append(emb.cpu().numpy())
        return (
            np.concatenate(out)
            if out
            else np.zeros((0, EMB_DIM), np.float32)
        )

    te = te.reset_index(drop=True)
    bg_te = bg_te.reset_index(drop=True)
    log("embedding test ships...")
    E = embed(te)
    ids = te["mmsi"].astype(str).to_numpy()
    sess = te["session"].astype(str).to_numpy()
    dates = te["date"].astype(str).to_numpy()
    log("embedding test background (for E3)...")
    BG = embed(bg_te) if len(bg_te) else np.zeros((0, EMB_DIM), np.float32)

    # MANDATORY: save BEFORE scoring
    torch.save(
        {
            "encoder": ecapa.state_dict(),
            "classifier": clf.state_dict(),
            "classes": classes,
            "config": {
                "protocol_label": "ECAPA-TDNN (fbank)",
                "objective": "AAM-softmax",
                "aam_margin": a.margin,
                "aam_scale": a.scale,
                "epochs": a.epochs,
                "bs": a.bs,
                "lr": a.lr,
                "emb_dim": EMB_DIM,
                "frontend": {
                    "sample_rate": SR_FBANK,
                    "n_mels": N_MELS,
                    "win_ms": WIN_MS,
                    "hop_ms": HOP_MS,
                    "n_fft": N_FFT,
                    "cmvn": "per-utterance mean subtract",
                    "cache_prefix": CACHE_PREFIX,
                },
                "ecapa_params": ecapa_n,
                "wall_clock_hours": round(wall_train, 3),
            },
        },
        CKPT_PATH,
    )
    np.savez(
        EMB_PATH,
        E=E,
        ids=ids,
        sess=sess,
        dates=dates,
        BG=BG,
    )
    log(f"saved weights -> {CKPT_PATH}")
    log(f"saved embeddings -> {EMB_PATH} E={E.shape} BG={BG.shape}")

    if a.skip_score:
        log("skip-score: stopping after save")
        return

    log("=== certified eval seeds 0..4 (mean + stats) ===")
    e2_1 = score_embeddings(E, ids)
    e2_1["params"] = ecapa_n
    e2_1["epochs"] = a.epochs
    e2_1["bs"] = a.bs
    e2_1["wall_clock_hours"] = round(wall_train, 3)
    e2_1["aam_margin"] = a.margin
    e2_1["aam_scale"] = a.scale
    e2_1["ecapa_fp32_mb"] = round(ecapa_n * 4 / 1e6, 3)
    e2_1["vs_ours_stats_pool"]["params_ratio"] = round(ecapa_n / OURS_PARAMS, 1)
    e2_1["commands"] = [
        r". .\scripts\use_tinyship.ps1",
        f"python scripts/reproduce/train_e2_ecapa.py --bs {a.bs} --epochs {a.epochs}",
        "python scripts/reproduce/score_e2_ecapa.py",
    ]
    merge_report(e2_1)
    mp, sp = e2_1["mean_pool"], e2_1["stats_pool"]
    log(
        f"RESULT mean_pool EER={mp['eer_mean']}±{mp['eer_std']} "
        f"top1={mp['top1_mean']}±{mp['top1_std']}"
    )
    log(
        f"RESULT stats_pool EER={sp['eer_mean']}±{sp['eer_std']} "
        f"top1={sp['top1_mean']}±{sp['top1_std']}"
    )
    log(
        f"vs ours stats {OURS_STATS_EER}: delta={e2_1['vs_ours_stats_pool']['delta']} "
        f"clip_disjoint={e2_1['clip_disjoint']}"
    )


if __name__ == "__main__":
    main()
