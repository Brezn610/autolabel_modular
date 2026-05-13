from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..settings.schemas import FrameRecord


def load_timesync_table(data_root: Path, max_frames: int) -> List[FrameRecord]:
    csv_path = data_root / "timesync_info.csv"
    xlsx_path = data_root / "timesync_info.xlsx"

    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.reader(f))
    elif xlsx_path.is_file():
        df = pd.read_excel(xlsx_path, header=None, engine="openpyxl")
        rows = df.astype(str).values.tolist()
    else:
        raise FileNotFoundError(f"未找到 {csv_path} 或 {xlsx_path}")

    if len(rows) < 3:
        raise ValueError("timesync 行数过少，无法解析")

    body = rows[1:]
    sensor_map: Dict[str, List[str]] = {}
    for r in body:
        if not r or not r[0]:
            continue
        sensor_map[r[0].strip()] = r[1:]

    if "middle_lidar" not in sensor_map:
        raise ValueError("timesync 中缺少 middle_lidar 行")

    n = min(len(v) for v in sensor_map.values())
    n = min(n, max_frames)
    frames: List[FrameRecord] = []
    for fi in range(n):
        ts = sensor_map.get("timestamp_nanoseconds", [None] * n)[fi] if "timestamp_nanoseconds" in sensor_map else None
        files = {k: vals[fi] for k, vals in sensor_map.items() if k != "timestamp_nanoseconds"}
        frames.append(FrameRecord(frame_index=fi, timestamp_ns=ts, files=files))
    return frames
