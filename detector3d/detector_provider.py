from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from .schema import VALID_CLASSES, VALID_SOURCES


class Detector3DProvider(ABC):
    @abstractmethod
    def predict(self) -> Dict[str, Any]:
        """Return canonical 3D detector output as a JSON-serializable dict."""


def validate_detections_json(doc: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(doc, dict):
        return ["top-level detections JSON must be an object"]

    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("top-level field 'metadata' must exist and be an object")
    else:
        for field in ("source", "coordinate_frame", "coordinate_frame_note", "warnings"):
            if field not in metadata:
                errors.append(f"metadata missing {field}")
        if metadata.get("source") is not None and metadata.get("source") not in VALID_SOURCES:
            errors.append(f"metadata.source must be one of {sorted(VALID_SOURCES)}, got {metadata.get('source')!r}")
        if "warnings" in metadata and not isinstance(metadata.get("warnings"), list):
            errors.append("metadata.warnings must be a list")

    frames = doc.get("frames")
    if not isinstance(frames, list):
        errors.append("top-level field 'frames' must exist and be a list")
        return errors

    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            errors.append(f"frames[{frame_idx}] must be an object")
            continue
        if "frame_id" not in frame:
            errors.append(f"frames[{frame_idx}] missing frame_id")
        if "boxes3d" not in frame:
            errors.append(f"frames[{frame_idx}] missing boxes3d")
            continue
        boxes = frame.get("boxes3d")
        if not isinstance(boxes, list):
            errors.append(f"frames[{frame_idx}].boxes3d must be a list")
            continue

        for box_idx, box in enumerate(boxes):
            prefix = f"frames[{frame_idx}].boxes3d[{box_idx}]"
            if not isinstance(box, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for field in ("det_id", "class", "box3d_lidar", "score_3d", "source"):
                if field not in box:
                    errors.append(f"{prefix} missing {field}")
            source = box.get("source")
            if source is not None and source not in VALID_SOURCES:
                errors.append(f"{prefix}.source must be one of {sorted(VALID_SOURCES)}, got {source!r}")
            cls = box.get("class")
            if cls is not None and cls not in VALID_CLASSES:
                errors.append(f"{prefix}.class must be one of {sorted(VALID_CLASSES)}, got {cls!r}")
            b = box.get("box3d_lidar")
            if not isinstance(b, list) or len(b) != 7:
                errors.append(f"{prefix}.box3d_lidar must be a list of length 7")
            else:
                try:
                    vals = [float(x) for x in b]
                    if vals[3] <= 0 or vals[4] <= 0 or vals[5] <= 0:
                        errors.append(f"{prefix}.box3d_lidar dimensions must be positive")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.box3d_lidar values must be numeric")
            try:
                score = float(box.get("score_3d"))
                if not (0.0 <= score <= 1.0):
                    errors.append(f"{prefix}.score_3d must be in [0, 1]")
            except (TypeError, ValueError):
                errors.append(f"{prefix}.score_3d must be numeric")
    return errors


def save_detections_json(doc: Dict[str, Any], path: Path) -> None:
    errors = validate_detections_json(doc)
    if errors:
        joined = "\n".join(errors[:20])
        raise ValueError(f"invalid detections JSON:\n{joined}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def load_detections_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    errors = validate_detections_json(doc)
    if errors:
        joined = "\n".join(errors[:20])
        raise ValueError(f"invalid detections JSON:\n{joined}")
    return doc
