from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import cv2

from dataset.annotation_projection_probe import DEFAULT_CAMERAS, _camera_image_path, _load_timesync
from .groundingdino_2d_schema import normalize_2d_class, save_2d_detections_json, validate_2d_detections_json


def convert_raw_dino_debug(input_path: Path, data_root: Path, max_frames: int = 0) -> Dict[str, Any]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    timesync = _load_timesync(data_root) if data_root else {}
    ts_by_index = list(sorted(timesync)) if timesync else []
    frames_out: List[Dict[str, Any]] = []
    per_class: Counter[str] = Counter()
    per_camera: Counter[str] = Counter()
    ignored: Counter[str] = Counter()

    for raw_frame in raw.get("frames", []):
        if max_frames and len(frames_out) >= max_frames:
            break
        frame_idx = int(raw_frame.get("frame_idx", len(frames_out)))
        timestamp = int(ts_by_index[frame_idx]) if frame_idx < len(ts_by_index) else frame_idx
        frame_id = f"{frame_idx:06d}"
        cameras_out: Dict[str, List[Dict[str, Any]]] = {}
        for camera, block in (raw_frame.get("cameras") or {}).items():
            detections_out: List[Dict[str, Any]] = []
            for det in block.get("detections") or []:
                cls = normalize_2d_class(det.get("label"))
                if cls is None:
                    ignored[str(det.get("label", ""))] += 1
                    continue
                det_id = f"2d_{frame_id}_{camera}_{len(detections_out):03d}"
                detections_out.append(
                    {
                        "det2d_id": det_id,
                        "class": cls,
                        "box2d": [float(x) for x in det.get("box_xyxy", [])],
                        "score_2d": float(det.get("score", 0.0)),
                        "source_label": str(det.get("label", "")),
                    }
                )
                per_class[cls] += 1
                per_camera[camera] += 1
            cameras_out[camera] = detections_out
        frames_out.append({"frame_id": frame_id, "timestamp": timestamp, "cameras": cameras_out})

    return {
        "metadata": {
            "source": "groundingdino",
            "classes": ["vehicle", "pedestrian"],
            "note": "2D detections used for semantic verification and 2D-3D matching",
            "converted_from": str(input_path),
            "summary": {
                "frames_processed": len(frames_out),
                "cameras_processed": sum(len(f["cameras"]) for f in frames_out),
                "total_2d_detections": sum(len(v) for f in frames_out for v in f["cameras"].values()),
                "per_class_counts": dict(per_class),
                "per_camera_counts": dict(per_camera),
                "ignored_label_counts": dict(ignored),
            },
        },
        "frames": frames_out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert existing raw GroundingDINO debug JSON to canonical 2D detections")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    doc = convert_raw_dino_debug(args.input, args.data_root, max_frames=args.max_frames)
    save_2d_detections_json(doc, args.output)
    errors = validate_2d_detections_json(doc)
    summary = doc["metadata"]["summary"]
    print("Converted GroundingDINO 2D detections")
    print(f"- frames processed: {summary['frames_processed']}")
    print(f"- cameras processed: {summary['cameras_processed']}")
    print(f"- total 2D detections: {summary['total_2d_detections']}")
    print(f"- per-class counts: {summary['per_class_counts']}")
    print(f"- per-camera counts: {summary['per_camera_counts']}")
    print(f"- validation errors: {len(errors)}")
    print(f"- output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
