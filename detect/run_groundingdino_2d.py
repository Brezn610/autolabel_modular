from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import cv2
import torch
from PIL import Image

from dataset.annotation_projection_probe import DEFAULT_CAMERAS, _camera_image_path, _load_timesync
from settings.dino_camera_config import dino_prompt_and_top_k

from .detection import grounding_dino_detect, load_grounding_dino, take_top_k_by_score
from .groundingdino_2d_schema import normalize_2d_class, save_2d_detections_json, validate_2d_detections_json


def _draw_debug(image_path: Path, detections: List[Dict[str, Any]], out_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    for det in detections:
        x1, y1, x2, y2 = [int(round(float(v))) for v in det["box2d"]]
        color = (0, 180, 255) if det["class"] == "vehicle" else (80, 220, 80)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{det['class']} {float(det['score_2d']):.2f}"
        cv2.putText(img, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def run_groundingdino_2d(
    *,
    data_root: Path,
    output: Path,
    max_frames: int,
    model_id: str,
    box_threshold: float,
    text_threshold: float,
    debug_images: bool,
    debug_dir: Path,
    max_debug_frames: int,
) -> Dict[str, Any]:
    timesync = _load_timesync(data_root)
    timestamps = sorted(timesync)[: int(max_frames)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_grounding_dino(model_id, device)

    frames_out: List[Dict[str, Any]] = []
    per_class: Counter[str] = Counter()
    per_camera: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    cameras = [c for c in DEFAULT_CAMERAS if (data_root / c).is_dir()]

    for frame_index, timestamp in enumerate(timestamps):
        frame_id = f"{frame_index:06d}"
        files = timesync[timestamp]
        cameras_out: Dict[str, List[Dict[str, Any]]] = {}
        for camera in cameras:
            image_name = files.get(camera)
            if not image_name:
                cameras_out[camera] = []
                continue
            image_path = _camera_image_path(data_root, camera, image_name)
            if not image_path.is_file():
                cameras_out[camera] = []
                continue
            pil = Image.open(image_path).convert("RGB")
            prompt, top_k = dino_prompt_and_top_k(camera)
            raw_dets = grounding_dino_detect(
                processor=processor,
                model=model,
                pil_image=pil,
                prompt=prompt,
                box_thr=float(box_threshold),
                text_thr=float(text_threshold),
                device=device,
            )
            raw_dets = take_top_k_by_score(raw_dets, top_k)
            detections_out: List[Dict[str, Any]] = []
            for det in raw_dets:
                cls = normalize_2d_class(det.get("label"))
                if cls is None:
                    ignored[str(det.get("label", ""))] += 1
                    continue
                det_id = f"2d_{frame_id}_{camera}_{len(detections_out):03d}"
                detections_out.append(
                    {
                        "det2d_id": det_id,
                        "class": cls,
                        "box2d": [float(v) for v in det["box_xyxy"]],
                        "score_2d": float(det["score"]),
                        "source_label": str(det.get("label", "")),
                    }
                )
                per_class[cls] += 1
                per_camera[camera] += 1
            cameras_out[camera] = detections_out
            if debug_images and frame_index < int(max_debug_frames):
                _draw_debug(image_path, detections_out, debug_dir / f"frame_{frame_id}__{camera}.jpg")
        frames_out.append({"frame_id": frame_id, "timestamp": int(timestamp), "cameras": cameras_out})

    doc = {
        "metadata": {
            "source": "groundingdino",
            "classes": ["vehicle", "pedestrian"],
            "note": "2D detections used for semantic verification and 2D-3D matching",
            "model_id": model_id,
            "box_threshold": float(box_threshold),
            "text_threshold": float(text_threshold),
            "summary": {
                "frames_processed": len(frames_out),
                "cameras_processed": sum(len(f["cameras"]) for f in frames_out),
                "total_2d_detections": sum(len(v) for f in frames_out for v in f["cameras"].values()),
                "per_class_counts": dict(per_class),
                "per_camera_counts": dict(per_camera),
                "ignored_label_counts": dict(ignored),
                "debug_image_dir": str(debug_dir) if debug_images else "",
            },
        },
        "frames": frames_out,
    }
    save_2d_detections_json(doc, output)
    return doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GroundingDINO for canonical 2D detections")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--model-id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box-threshold", type=float, default=0.01)
    parser.add_argument("--text-threshold", type=float, default=0.01)
    parser.add_argument("--debug-images", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/debug/groundingdino_2d_probe"))
    parser.add_argument("--max-debug-frames", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    doc = run_groundingdino_2d(
        data_root=args.data_root,
        output=args.output,
        max_frames=args.max_frames,
        model_id=args.model_id,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        debug_images=args.debug_images,
        debug_dir=args.debug_dir,
        max_debug_frames=args.max_debug_frames,
    )
    errors = validate_2d_detections_json(doc)
    summary = doc["metadata"]["summary"]
    print("GroundingDINO 2D detection summary")
    print(f"- frames processed: {summary['frames_processed']}")
    print(f"- cameras processed: {summary['cameras_processed']}")
    print(f"- total 2D detections: {summary['total_2d_detections']}")
    print(f"- per-class counts: {summary['per_class_counts']}")
    print(f"- per-camera counts: {summary['per_camera_counts']}")
    print(f"- validation errors: {len(errors)}")
    print(f"- output: {args.output}")
    if summary.get("debug_image_dir"):
        print(f"- debug images: {summary['debug_image_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
