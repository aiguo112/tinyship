"""Attach real 1 s VTUAD wavs (base64) onto TSMR_DATA inside tsmr_demo.html."""
from __future__ import annotations

import base64
import io
import json
import re
import struct
import wave
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "tsmr_demo.html"
SPLIT = ROOT / "data" / "processed" / "splits" / "mmsi_test.csv"
TARGET_SR = 16000


def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"not a WAVE file: {path}")
    pos = 12
    fmt = None
    audio = None
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        sz = struct.unpack_from("<I", raw, pos + 4)[0]
        chunk = raw[pos + 8 : pos + 8 + sz]
        if cid == b"fmt ":
            tag, ch, sr, _br, _ba, bps = struct.unpack_from("<HHIIHH", chunk)
            fmt = (tag, ch, sr, bps)
        elif cid == b"data":
            audio = chunk
        pos += 8 + sz + (sz % 2)
        if cid == b"data":
            break
    if fmt is None or audio is None:
        raise ValueError(f"missing fmt/data: {path}")
    tag, ch, sr, bps = fmt
    if tag == 3 and bps == 32:
        y = np.frombuffer(audio, dtype="<f4").astype(np.float32, copy=True)
    elif tag == 1 and bps == 16:
        y = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bps == 32:
        y = np.frombuffer(audio, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav fmt={fmt} for {path}")
    if ch > 1:
        y = y.reshape(-1, ch).mean(axis=1).astype(np.float32)
    return y, int(sr)


def wav_data_uri(path: str) -> str:
    y, sr = read_wav_mono(path)
    if sr != TARGET_SR and len(y) > 1:
        n_new = max(8, int(round(len(y) * TARGET_SR / float(sr))))
        y = np.interp(
            np.linspace(0, len(y) - 1, n_new),
            np.arange(len(y)),
            y,
        ).astype(np.float32)
        sr = TARGET_SR
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = np.clip(y / peak * 0.88, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:audio/wav;base64," + b64


def main():
    html = HTML.read_text(encoding="utf-8")
    m = re.search(
        r'<script id="tsmr-data">window\.TSMR_DATA=(.*);</script>',
        html,
        flags=re.S,
    )
    if not m:
        raise SystemExit("TSMR_DATA blob not found in tsmr_demo.html")
    data = json.loads(m.group(1))

    df = pd.read_csv(SPLIT)
    df["MMSI"] = df["MMSI"].astype(str)
    df["name"] = df["file_path"].map(lambda p: Path(str(p)).name)

    n_ok, n_miss = 0, 0
    for v in data["vessels"]:
        sub = df[df["MMSI"] == str(v["mmsi"])]
        for probe in v["probes"]:
            hit = sub[sub["name"] == probe["file"]]
            if hit.empty:
                n_miss += 1
                probe["audio"] = ""
                continue
            path = str(hit.iloc[0]["file_path"])
            if not Path(path).exists():
                n_miss += 1
                probe["audio"] = ""
                continue
            probe["audio"] = wav_data_uri(path)
            n_ok += 1

    blob = '<script id="tsmr-data">window.TSMR_DATA=' + json.dumps(
        data, separators=(",", ":")
    ) + ";</script>"
    html = html[: m.start()] + blob + html[m.end() :]
    HTML.write_text(html, encoding="utf-8")
    print(f"audio clips ok={n_ok} miss={n_miss} html={HTML.stat().st_size}")


if __name__ == "__main__":
    main()
