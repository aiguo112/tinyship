#!/usr/bin/env python
"""E2.0 ECAPA-TDNN feasibility only — dry forward + step timing. NO full train."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

# torchaudio 2.11+ removed list_audio_backends; SpeechBrain 1.0.3 still calls it.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier
from speechbrain.nnet.losses import AdditiveAngularMargin, LogSoftmaxWrapper

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.fingerprint import build_encoder  # noqa: E402

REPORTS = ROOT / "reports"
SPLIT = ROOT / "data" / "processed" / "splits" / "mmsi_train.csv"

# SpeechBrain / VoxCeleb ECAPA-style fbank (Desplanques et al. 2020 defaults)
SR_FBANK = 16000
N_MELS = 80
WIN_MS = 25
HOP_MS = 10
N_FFT = 400
HOP_LENGTH = int(SR_FBANK * HOP_MS / 1000)  # 160
WIN_LENGTH = int(SR_FBANK * WIN_MS / 1000)  # 400
EMB_DIM = 192
N_CLASSES = 178


def nparams(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def load_wav_1s(path: str, target_sr: int = SR_FBANK) -> torch.Tensor:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    wav = torch.from_numpy(y)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
    n = target_sr
    if wav.numel() > n:
        wav = wav[:n]
    elif wav.numel() < n:
        wav = F.pad(wav, (0, n - wav.numel()))
    return wav


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} name={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")

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
    ).to(device)
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)

    def fbank_feats(wav_b: torch.Tensor) -> torch.Tensor:
        m = to_db(mel(wav_b))  # (B, n_mels, frames)
        m = m - m.mean(dim=-1, keepdim=True)
        return m.transpose(1, 2)  # (B, T, F) for ECAPA

    tr = pd.read_csv(SPLIT)
    tr = tr[tr.MMSI.astype(str) != "0"]
    n_train = len(tr)
    paths = tr.file_path.astype(str).head(8).tolist()
    wavs = torch.stack([load_wav_1s(p) for p in paths]).to(device)
    feats = fbank_feats(wavs)
    frames_1s = int(feats.shape[1])
    print(
        f"frontend=fbank sr={SR_FBANK} n_mels={N_MELS} "
        f"win={WIN_MS}ms hop={HOP_MS}ms frames/1s={frames_1s}"
    )
    print(f"wav={tuple(wavs.shape)} feats={tuple(feats.shape)}")

    ecapa = ECAPA_TDNN(input_size=N_MELS, lin_neurons=EMB_DIM).to(device)
    lengths = torch.ones(feats.size(0), device=device)
    with torch.no_grad():
        emb = ecapa(feats, lengths=lengths)
    print(f"emb_shape={tuple(emb.shape)}")

    ours = build_encoder(2, 128, 0.25)
    ecapa_n = nparams(ecapa)
    ours_n = nparams(ours)
    print(f"ecapa_params={ecapa_n} ({ecapa_n/1e6:.3f}M) fp32_MB={ecapa_n*4/1e6:.2f}")
    print(f"ours_params={ours_n} ({ours_n/1e6:.3f}M) fp32_MB={ours_n*4/1e6:.2f}")
    print(f"ratio={ecapa_n/ours_n:.1f}")

    clf = Classifier(input_size=EMB_DIM, lin_neurons=EMB_DIM, out_neurons=N_CLASSES).to(device)
    aam = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.2, scale=30)).to(device)
    print(f"classifier_params={nparams(clf)}")

    ecapa.train()
    clf.train()
    opt = torch.optim.Adam(list(ecapa.parameters()) + list(clf.parameters()), lr=1e-3)

    def timed_steps(bs: int, n_steps: int, warmup: int = 3) -> float:
        wav_b = wavs[:1].repeat(bs, 1)
        feats_b = fbank_feats(wav_b)
        labels = torch.arange(bs, device=device) % N_CLASSES
        lens = torch.ones(bs, device=device)
        for _ in range(warmup):
            opt.zero_grad(set_to_none=True)
            e = ecapa(feats_b, lengths=lens)
            e_in = e.unsqueeze(1) if e.dim() == 2 else e
            loss = aam(clf(e_in), labels.unsqueeze(1))
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            e = ecapa(feats_b, lengths=lens)
            e_in = e.unsqueeze(1) if e.dim() == 2 else e
            loss = aam(clf(e_in), labels.unsqueeze(1))
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_steps

    sec8 = timed_steps(8, 40, warmup=5)
    sec32 = timed_steps(32, 30)
    sec128 = timed_steps(128, 20)
    print(f"sec_per_step bs8={sec8:.4f} bs32={sec32:.4f} bs128={sec128:.4f}")

    epochs = 120
    steps32 = int(np.ceil(n_train / 32))
    steps128 = int(np.ceil(n_train / 128))
    hours32 = steps32 * epochs * sec32 / 3600
    hours128 = steps128 * epochs * sec128 / 3600
    print(f"n_train={n_train} steps/ep bs32={steps32} bs128={steps128}")
    print(f"est_hours epochs={epochs} bs32={hours32:.2f} bs128={hours128:.2f}")

    max_mem = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None
    print(f"max_mem_MB={max_mem}")

    out = {
        "experiment": "E2.0",
        "status": "feasibility_only",
        "implementation": (
            "SpeechBrain 1.0.3 ECAPA_TDNN (Desplanques et al. 2020); "
            "torchaudio 2.11 compat shim: list_audio_backends -> ['soundfile']"
        ),
        "frontend": {
            "choice": "fbank",
            "details": {
                "sample_rate": SR_FBANK,
                "n_mels": N_MELS,
                "win_ms": WIN_MS,
                "hop_ms": HOP_MS,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "win_length": WIN_LENGTH,
                "frames_per_1s": frames_1s,
                "cmvn": "per-utterance mean subtract",
                "source_audio": "VTUAD 1s clips @ 32kHz resampled to 16kHz",
                "note": (
                    "Native ECAPA/SpeechBrain-style log-mel fbank, NOT our CQT+MFCC 2-ch. "
                    "Labeled as a separate comparison row."
                ),
            },
            "protocol_label": "ECAPA-TDNN (fbank)",
        },
        "params": {
            "ecapa": ecapa_n,
            "ecapa_fp32_mb": round(ecapa_n * 4 / 1e6, 3),
            "classifier_aam_178": nparams(clf),
            "ours_mobilenet": ours_n,
            "ours_fp32_mb": round(ours_n * 4 / 1e6, 3),
            "ratio": round(ecapa_n / ours_n, 2),
        },
        "dry_forward": {
            "emb_dim": EMB_DIM,
            "input_shape": list(feats.shape),
            "emb_shape": list(emb.shape),
            "ok": True,
        },
        "cost_estimate": {
            "seconds_per_step": round(sec32, 5),
            "seconds_per_step_bs8": round(sec8, 5),
            "seconds_per_step_bs128": round(sec128, 5),
            "batch_size_assumed": 32,
            "est_full_train_hours": round(hours32, 3),
            "est_full_train_hours_bs128": round(hours128, 3),
            "assumed_epochs": epochs,
            "assumed_steps_per_epoch": steps32,
            "n_train_clips": int(n_train),
            "max_cuda_mem_mb": round(max_mem, 1) if max_mem is not None else None,
            "note": (
                "Timed forward+backward+Adam on synthetic batch of real-clip fbank "
                "(feature extract outside timed loop for bs>8 reuse). "
                "Excludes dataloader/IO overhead; real wall-clock may be ~1.2–1.5x."
            ),
        },
        "e2_1_plan": {
            "split": "178 train / 22 unseen test MMSI-disjoint (data/processed/splits/mmsi_*.csv)",
            "frontend": "fbank 16kHz 80-mel win25ms hop10ms; label row ECAPA-TDNN (fbank)",
            "objective": (
                "SpeechBrain default: Additive Angular Margin (AAM-softmax) via "
                "LogSoftmaxWrapper(AdditiveAngularMargin); NOT our SubCenter ArcFace (K=3). "
                "Document AAM vs SubCenter in table notes."
            ),
            "encoder": "ECAPA_TDNN(input_size=80, lin_neurons=192) ~6M params",
            "eval": "certified_verify.py k_enroll=20 probe_w=3 seeds 0-4; mean±std",
            "save_before_score": "weights + embeddings before scoring (mandatory)",
            "epochs_bs": f"{epochs} epochs, prefer bs=32 (or 128 if stable); Adam/AdamW",
        },
        "commands": [
            r". .\scripts\use_tinyship.ps1",
            r'pip install "speechbrain>=1.0,<1.1"',
            r"python scripts/reproduce/e2_0_ecapa_feasibility.py",
        ],
        "blocker": None,
        "ready_for_e2_1": True,
        "speechbrain_version": "1.0.3",
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / "e2_ecapa.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")
    return out


if __name__ == "__main__":
    main()
