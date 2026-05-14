from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_2D_CLASSES = {"vehicle", "pedestrian"}


def normalize_2d_class(label: Any) -> Optional[str]:
    text = str(label or "").strip().lower().replace(".", " ")
    tokens = set(text.split())
    if tokens & {"car", "truck", "bus", "van", "vehicle", "suv", "sedan", "pickup"}:
        return "vehicle"
    if tokens & {"pedestrian", "person", "people"}:
        return "pedestrian"
    return None


def validate_2d_detections_json(doc: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(doc, dict):
        return ["top-level JSON must be an object"]
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must exist and be an object")
    frames = doc.get("frames")
    if not isinstance(frames, list):
        errors.append("frames must exist and be a list")
        return errors
    for frame_idx, frame in enumerate(frames):
        prefix = f"frames[{frame_idx}]"
        for field in ("frame_id", "timestamp", "cameras"):
            if field not in frame:
                errors.append(f"{prefix} missing {field}")
        cameras = frame.get("cameras")
        if not isinstance(cameras, dict):
            errors.append(f"{prefix}.cameras must be an object")
            continue
        for camera, detections in cameras.items():
            if not isinstance(detections, list):
                errors.append(f"{prefix}.cameras.{camera} must be a list")
                continue
            for det_idx, det in enumerate(detections):
                dp = f"{prefix}.cameras.{camera}[{det_idx}]"
                if not isinstance(det, dict):
                    errors.append(f"{dp} must be an object")
                    continue
                for field in ("det2d_id", "class", "box2d", "score_2d"):
                    if field not in det:
                        errors.append(f"{dp} missing {field}")
                if det.get("class") not in VALID_2D_CLASSES:
                    errors.append(f"{dp}.class must be one of {sorted(VALID_2D_CLASSES)}")
                box = det.get("box2d")
                if not isinstance(box, list) or len(box) != 4:
                    errors.append(f"{dp}.box2d must be a list of 4 numbers")
                else:
                    try:
                        [float(x) for x in box]
                    except (TypeError, ValueError):
                        errors.append(f"{dp}.box2d values must be numeric")
                try:
                    score = float(det.get("score_2d"))
                    if not (0.0 <= score <= 1.0):
                        errors.append(f"{dp}.score_2d must be in [0, 1]")
                except (TypeError, ValueError):
                    errors.append(f"{dp}.score_2d must be numeric")
    return errors


def save_2d_detections_json(doc: Dict[str, Any], path: Path) -> None:
    errors = validate_2d_detections_json(doc)
    if errors:
        raise ValueError("invalid 2D detections JSON:\n" + "\n".join(errors[:20]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

