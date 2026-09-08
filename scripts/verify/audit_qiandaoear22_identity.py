#!/usr/bin/env python
"""QiandaoEar22 identity feasibility audit — no training."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "qiandaoear22"
MANIFEST = ROOT / "data" / "metadata" / "qiandaoear22_manifest.csv"
STATUS = ROOT / "data" / "metadata" / "qiandaoear22_download_status.json"
REPORTS = ROOT / "reports"

PAT = re.compile(
    r"^(?P<ts>\d{14})_(?P<sm>[SM])_(?P<body>.+?)(?:_label__\d+__\d+)?$",
    re.I,
)


def parse_name(fn: str):
    stem = Path(fn).stem
    stem = re.sub(r"\s*-.*$", "", stem)  # drop ' - 副本' etc
    mobj = PAT.match(stem)
    if not mobj:
        return None
    ts, sm, body = mobj.group("ts"), mobj.group("sm").upper(), mobj.group("body")
    body = re.sub(r"_0$", "", body)
    targets = [t.strip() for t in re.split(r"\s*&\s*", body) if t.strip()]
    ships = []
    for t in targets:
        parts = [p for p in t.split("_") if p != ""]
        dist = aud = None
        ship_parts = parts[:]
        if len(parts) >= 3 and parts[1] in "NMF" and parts[2] in "SMW":
            ship_parts = [parts[0]]
            dist, aud = parts[1], parts[2]
        elif len(parts) >= 3 and parts[-2] in "NMF" and parts[-1] in "SMW":
            dist, aud = parts[-2], parts[-1]
            ship_parts = parts[:-2]
        ship = "_".join(ship_parts) if ship_parts else t
        ships.append((ship, dist, aud))
    return {
        "timestamp": ts,
        "date": ts[:8],
        "sm": sm,
        "ships": ships,
        "n_targets": len(ships),
        "stem": stem,
    }


def occasion_stats(shipdf: pd.DataFrame, key_cols: list[str]):
    g = shipdf.groupby(["ship"] + key_cols).size().reset_index(name="n_clips")
    strong = g[g.n_clips >= 20]
    per_ship = strong.groupby("ship").size().rename("n_occasions_ge20")
    any_occ = g.groupby("ship").size().rename("n_occasions_any")
    ships = sorted(shipdf["ship"].dropna().unique())
    tab = pd.DataFrame({"ship": ships}).set_index("ship")
    tab = tab.join(any_occ).join(per_ship)
    tab["n_occasions_any"] = tab["n_occasions_any"].fillna(0).astype(int)
    tab["n_occasions_ge20"] = tab["n_occasions_ge20"].fillna(0).astype(int)
    gate = int((tab["n_occasions_ge20"] >= 2).sum())
    return gate, tab, g


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    wavs = list(RAW.rglob("*.wav")) + list(RAW.rglob("*.WAV"))
    manifest = pd.read_csv(MANIFEST)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}

    m = manifest.dropna(subset=["sha256"]).drop_duplicates("sha256").copy()

    rows, fails = [], []
    for _, r in m.iterrows():
        fn = (
            str(r["original_filename"])
            if pd.notna(r.get("original_filename"))
            else Path(str(r["file_path"])).name
        )
        parsed = parse_name(fn)
        top = Path(str(r["file_path"])).parts
        try:
            idx = next(i for i, p in enumerate(top) if p.lower() == "qiandaoear22")
            folder = top[idx + 1] if idx + 1 < len(top) else ""
        except StopIteration:
            folder = ""
        if not parsed:
            fails.append(fn)
            rows.append(
                dict(
                    sha256=r["sha256"],
                    file_path=r["file_path"],
                    folder=folder,
                    kind="unparsed",
                    filename=fn,
                )
            )
            continue
        for i, (ship, dist, aud) in enumerate(parsed["ships"]):
            rows.append(
                dict(
                    sha256=r["sha256"],
                    file_path=r["file_path"],
                    folder=folder,
                    kind="ship",
                    filename=fn,
                    timestamp=parsed["timestamp"],
                    date=parsed["date"],
                    single_multi=parsed["sm"],
                    ship=ship,
                    distance=dist,
                    audibility=aud,
                    n_targets=parsed["n_targets"],
                    target_idx=i,
                )
            )

    df = pd.DataFrame(rows)
    shipdf = df[df.kind == "ship"].copy()
    file_level = shipdf.sort_values("target_idx").drop_duplicates("sha256")

    results = {}
    for label, cols in [
        ("by_date", ["date"]),
        ("by_distance", ["distance"]),
        ("by_audibility", ["audibility"]),
        ("by_date_distance", ["date", "distance"]),
        ("by_date_audibility", ["date", "audibility"]),
        ("by_distance_audibility", ["distance", "audibility"]),
    ]:
        gate, tab, _ = occasion_stats(shipdf, cols)
        results[label] = {
            "gate_vessels_ge2_occasions_ge20clips": gate,
            "vessels_total": int(len(tab)),
            "occasion_count_hist": {
                int(k): int(v)
                for k, v in tab["n_occasions_ge20"].value_counts().sort_index().items()
            },
            "vessels_with_ge2_any_occasion": int((tab["n_occasions_any"] >= 2).sum()),
            "per_ship": tab.reset_index().to_dict("records"),
        }

    bout = shipdf.groupby(["ship", "timestamp"]).size().reset_index(name="n")
    bout20 = bout[bout.n >= 20]
    by_ts_gate = int((bout20.groupby("ship").size() >= 2).sum())

    # single-target only
    single = shipdf[shipdf["single_multi"] == "S"]
    single_gates = {}
    for label, cols in [
        ("by_date", ["date"]),
        ("by_distance", ["distance"]),
        ("by_audibility", ["audibility"]),
        ("by_timestamp", ["timestamp"]),
    ]:
        if single.empty:
            single_gates[label] = 0
            continue
        gate, _, _ = occasion_stats(single, cols)
        single_gates[label] = gate

    srs: Counter = Counter()
    step = max(1, len(file_level) // 50)
    for fp in file_level["file_path"].iloc[::step]:
        try:
            srs[sf.info(str(fp)).samplerate] += 1
        except Exception:
            pass

    # clip counts per ship overall
    clips_per_ship = (
        shipdf.drop_duplicates(["sha256", "ship"])
        .groupby("ship")
        .size()
        .sort_values(ascending=False)
    )

    archive_count = len(status.get("archives_found", []))
    payload = {
        "availability": {
            "local_wav_paths_on_disk": len(wavs),
            "manifest_rows": int(len(manifest)),
            "unique_sha256": int(m["sha256"].nunique()),
            "archives_listed_in_status": archive_count,
            "status_completeness": status.get("completeness"),
            "official_github": status.get("official_github"),
            "ieee_dataport": status.get("ieee_dataport"),
            "onedrive": status.get("onedrive_readme_url"),
            "note": (
                "Local set is present (IEEE/OneDrive archives extracted). "
                "Disk WAV count >> unique SHA256 due to duplicate extracts / 副本 copies. "
                "Paper reports 10,611 ship + 25,900 noise records; local unique content is compared below."
            ),
        },
        "identity": {
            "unique_vessel_name_labels": int(shipdf["ship"].nunique()),
            "vessel_names": sorted(shipdf["ship"].dropna().unique().tolist()),
            "clips_per_vessel_name": {k: int(v) for k, v in clips_per_ship.items()},
            "unique_ship_labeled_files": int(file_level["sha256"].nunique()),
            "dates": sorted(file_level["date"].unique().tolist()) if len(file_level) else [],
            "n_dates": int(file_level["date"].nunique()) if len(file_level) else 0,
            "n_timestamps": int(shipdf["timestamp"].nunique()) if len(shipdf) else 0,
            "distance_counts": {
                str(k): int(v) for k, v in shipdf["distance"].value_counts(dropna=False).items()
            },
            "audibility_counts": {
                str(k): int(v)
                for k, v in shipdf["audibility"].value_counts(dropna=False).items()
            },
            "single_multi_file_counts": {
                str(k): int(v)
                for k, v in file_level["single_multi"].value_counts().items()
            },
            "sample_rates_observed": {str(k): int(v) for k, v in srs.items()},
            "paper_claim": (
                "Filenames encode vessel name + distance(N/M/F) + audibility(S/M/W) + "
                "single/multi (S/M). Paper: ~20 ship-target categories; NOT AIS/MMSI IDs."
            ),
        },
        "gates_all_targets": results,
        "gate_by_timestamp_ge2_occ_ge20": by_ts_gate,
        "gates_single_target_only": single_gates,
        "unparsed_count": int((df.kind == "unparsed").sum()),
        "unparsed_examples": fails[:20],
    }
    (REPORTS / "qiandaoear22_identity_data.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "unique_sha256": payload["availability"]["unique_sha256"],
        "vessel_names": payload["identity"]["vessel_names"],
        "n_vessels": payload["identity"]["unique_vessel_name_labels"],
        "dates": payload["identity"]["dates"],
        "gates": {k: v["gate_vessels_ge2_occasions_ge20clips"] for k, v in results.items()},
        "by_timestamp_gate": by_ts_gate,
        "single_gates": single_gates,
        "unparsed": payload["unparsed_count"],
        "sample_rates": payload["identity"]["sample_rates_observed"],
        "clips_per_vessel": payload["identity"]["clips_per_vessel_name"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
