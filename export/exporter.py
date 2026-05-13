from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..settings.schemas import FrameRecord


def build_annotations_export(
    frames: List[FrameRecord],
    per_frame_objects: List[List[Dict[str, Any]]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "metadata": metadata,
        "frames": [
            {
                "frame_idx": fr.frame_index,
                "timestamp_ns": fr.timestamp_ns,
                "lidar_file": fr.files.get("middle_lidar"),
                "objects": objs,
            }
            for fr, objs in zip(frames, per_frame_objects)
        ],
    }


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
