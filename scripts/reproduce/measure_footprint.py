#!/usr/bin/env python
"""
Honest footprint measurements for MobileNet encoder + SpeechBrain ECAPA_TDNN.

  python scripts/reproduce/measure_footprint.py

No fabricated energy. Checkpoint bytes labeled as fp32 weight proxy, not MCU flash.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.fingerprint import build_encoder  # noqa: E402

REPORTS = ROOT / "reports"
CACHE = ROOT / "data" / "processed" / "feat_cache"
N_WARMUP = 20
N_TIMED = 100


def n_trainable(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def state_dict_bytes(m: torch.nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return int(buf.tell())


def time_forwards(model, x, device: str, n_warmup=N_WARMUP, n_timed=N_TIMED):
    model = model.to(device).eval()
    x = x.to(device)
    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x) if device == "cpu" or not hasattr(model, "forward") else _fwd(model, x)
            if device == "cuda":
                torch.cuda.synchronize()
    times = []
    peak_mb = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in range(n_timed):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = _fwd(model, x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    if device == "cuda":
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 3)
    arr = np.asarray(times, dtype=float)
    return {
        "ms_mean": round(float(arr.mean()), 4),
        "ms_std": round(float(arr.std(ddof=1)), 4),
        "n_warmup": n_warmup,
        "n_timed": n_timed,
        "peak_ram_mb": peak_mb,
    }


def _fwd(model, x):
    # ECAPA expects (B, T, F) + lengths; MobileNet expects (B, C, H, W)
    if hasattr(model, "_tinyship_is_ecapa"):
        lengths = torch.ones(x.shape[0], device=x.device)
        return model(x, lengths=lengths)
    return model(x)


def cpu_rss_optional(model, x) -> dict:
    try:
        import psutil
        import os

        proc = psutil.Process(os.getpid())
        rss0 = proc.memory_info().rss
        model = model.to("cpu").eval()
        x = x.to("cpu")
        with torch.no_grad():
            for _ in range(5):
                _ = _fwd(model, x)
        rss1 = proc.memory_info().rss
        return {
            "measured": True,
            "rss_delta_mb": round((rss1 - rss0) / (1024**2), 3),
            "rss_after_mb": round(rss1 / (1024**2), 3),
            "note": "process RSS delta around CPU forwards (coarse; not peak allocator)",
        }
    except ImportError:
        return {
            "measured": False,
            "note": "psutil not available — CPU peak RAM skipped",
        }


def real_mobilenet_input():
    # Prefer a real cached feature
    files = sorted(CACHE.glob("cqt_mfcc_*.npy"))
    if files:
        feat = np.load(files[0])
        assert feat.ndim == 3 and feat.shape[0] == 2 and feat.shape[1] == 95, feat.shape
        x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0)  # (1,2,95,T)
        return x, {
            "source": str(files[0].relative_to(ROOT)).replace("\\", "/"),
            "shape": list(x.shape),
        }
    x = torch.randn(1, 2, 95, 256)
    return x, {"source": "synthetic", "shape": list(x.shape), "note": "no cache found; T=256"}


def measure_mobilenet():
    enc = build_encoder(2, 128, 0.25)
    # load trained weights if present (footprint of deployed encoder)
    ckpt = REPORTS / "fingerprint_attention_ckpt.pt"
    weight_source = "fresh_init"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "encoder" in ck:
            enc.load_state_dict(ck["encoder"])
            weight_source = "fingerprint_attention_ckpt.pt encoder"
    x, meta = real_mobilenet_input()
    params = n_trainable(enc)
    disk = state_dict_bytes(enc)

    out = {
        "name": "MobileNetV3-Small 0.25 (TinyShip encoder)",
        "build": "build_encoder(2,128,0.25)",
        "weight_source": weight_source,
        "input": meta,
        "params_trainable": params,
        "checkpoint_bytes_fp32": disk,
        "checkpoint_mb_fp32": round(disk / (1024**2), 4),
        "checkpoint_note": (
            "checkpoint bytes (fp32 weights), not MCU flash after quantization"
        ),
        "emb_dim": 128,
        "energy_j": None,
        "energy_note": "not measured (no power meter)",
    }
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        out["latency_cuda"] = time_forwards(enc, x, "cuda")
        out["peak_ram_cuda_mb"] = out["latency_cuda"]["peak_ram_mb"]
        out["cuda_device"] = torch.cuda.get_device_name(0)
    else:
        out["latency_cuda"] = None
        out["peak_ram_cuda_mb"] = None
        out["cuda_note"] = "CUDA not available"
    out["latency_cpu"] = time_forwards(enc, x, "cpu")
    out["peak_ram_cpu"] = cpu_rss_optional(enc, x)
    return out


def measure_ecapa():
    # SpeechBrain shim (same as train_e2)
    try:
        import torchaudio

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
    except Exception:
        pass

    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

    ecapa = ECAPA_TDNN(input_size=80, lin_neurons=192)
    ecapa._tinyship_is_ecapa = True
    weight_source = "fresh_init matching e2 (input_size=80, lin_neurons=192)"
    ckpt = REPORTS / "e2_ecapa_ckpt.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ecapa.load_state_dict(ck["encoder"])
        weight_source = "e2_ecapa_ckpt.pt encoder"
        # also report on-disk encoder-only serialization
        enc_bytes_from_ckpt = None
        buf = io.BytesIO()
        torch.save(ck["encoder"], buf)
        enc_bytes_from_ckpt = int(buf.tell())
    else:
        enc_bytes_from_ckpt = None

    x = torch.randn(1, 101, 80)  # (B, T, F) fbank
    params = n_trainable(ecapa)
    disk = state_dict_bytes(ecapa)

    out = {
        "name": "SpeechBrain ECAPA_TDNN",
        "build": "ECAPA_TDNN(input_size=80, lin_neurons=192)",
        "weight_source": weight_source,
        "input": {"source": "synthetic_fbank_shape", "shape": list(x.shape)},
        "params_trainable": params,
        "checkpoint_bytes_fp32": disk,
        "checkpoint_mb_fp32": round(disk / (1024**2), 4),
        "checkpoint_bytes_from_ckpt_encoder": enc_bytes_from_ckpt,
        "checkpoint_note": (
            "checkpoint bytes (fp32 weights), not MCU flash after quantization"
        ),
        "emb_dim": 192,
        "frontend_note": "Native 16 kHz log-mel fbank (not CQT+MFCC)",
        "energy_j": None,
        "energy_note": "not measured (no power meter)",
    }
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        out["latency_cuda"] = time_forwards(ecapa, x, "cuda")
        out["peak_ram_cuda_mb"] = out["latency_cuda"]["peak_ram_mb"]
        out["cuda_device"] = torch.cuda.get_device_name(0)
    else:
        out["latency_cuda"] = None
        out["peak_ram_cuda_mb"] = None
        out["cuda_note"] = "CUDA not available"
    out["latency_cpu"] = time_forwards(ecapa, x, "cpu")
    out["peak_ram_cpu"] = cpu_rss_optional(ecapa, x)
    return out


def main():
    print("=== FOOTPRINT: MobileNet ===")
    mob = measure_mobilenet()
    print(
        f"  params={mob['params_trainable']} disk={mob['checkpoint_bytes_fp32']} B "
        f"({mob['checkpoint_mb_fp32']} MB)"
    )
    if mob.get("latency_cuda"):
        print(
            f"  CUDA {mob['latency_cuda']['ms_mean']:.3f}±{mob['latency_cuda']['ms_std']:.3f} ms/clip "
            f"peak={mob['peak_ram_cuda_mb']} MB"
        )
    print(
        f"  CPU  {mob['latency_cpu']['ms_mean']:.3f}±{mob['latency_cpu']['ms_std']:.3f} ms/clip"
    )

    print("=== FOOTPRINT: ECAPA ===")
    ecapa = measure_ecapa()
    print(
        f"  params={ecapa['params_trainable']} disk={ecapa['checkpoint_bytes_fp32']} B "
        f"({ecapa['checkpoint_mb_fp32']} MB)"
    )
    if ecapa.get("latency_cuda"):
        print(
            f"  CUDA {ecapa['latency_cuda']['ms_mean']:.3f}±{ecapa['latency_cuda']['ms_std']:.3f} ms/clip "
            f"peak={ecapa['peak_ram_cuda_mb']} MB"
        )
    print(
        f"  CPU  {ecapa['latency_cpu']['ms_mean']:.3f}±{ecapa['latency_cpu']['ms_std']:.3f} ms/clip"
    )

    out = {
        "experiment": "footprint",
        "energy_j": None,
        "energy_note": "not measured (no power meter)",
        "models": {
            "mobilenet": mob,
            "ecapa": ecapa,
        },
        "honesty": [
            "Disk/flash proxy = serialized fp32 state_dict bytes, not post-quant MCU flash.",
            "Latency = batch=1 forward only (no feature extract / dataloader).",
            "CUDA peak RAM = torch.cuda.max_memory_allocated during timed forwards.",
            "Energy not measured.",
        ],
    }
    op = REPORTS / "footprint.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
