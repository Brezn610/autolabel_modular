from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataset.driving_annotations import load_driving_annotations, normalize_class

from .detector_provider import Detector3DProvider, load_detections_json, save_detections_json, validate_detections_json
from .schema import COORDINATE_FRAME_NOTE, Detection3D, Detections3DDocument, FrameDetections3D


def _wrap_yaw_0_2pi(yaw: float) -> float:
    return float(yaw % (2.0 * math.pi))


def _distance_adaptive_xy_std(x: float, y: float, default_xy_std: float) -> float:
    distance = math.hypot(float(x), float(y))
    if distance < 20.0:
        return min(float(default_xy_std), 0.20)
    if distance < 40.0:
        return max(float(default_xy_std), 0.35)
    return max(float(default_xy_std), 0.50)


class MockLidarDetectorProvider(Detector3DProvider):
    def __init__(
        self,
        *,
        annotations: Path,
        seed: int = 42,
        max_frames: Optional[int] = None,
        vehicle_keep_prob: float = 0.85,
        pedestrian_keep_prob: float = 0.65,
        xy_std: float = 0.30,
        z_std: float = 0.10,
        yaw_std: float = 0.08,
        include_debug_fields: bool = False,
    ) -> None:
        self.annotations = annotations
        self.rng = random.Random(int(seed))
        self.seed = int(seed)
        self.max_frames = max_frames
        self.vehicle_keep_prob = float(vehicle_keep_prob)
        self.pedestrian_keep_prob = float(pedestrian_keep_prob)
        self.xy_std = float(xy_std)
        self.z_std = float(z_std)
        self.yaw_std = float(yaw_std)
        self.include_debug_fields = bool(include_debug_fields)
        self.summary: Dict[str, Any] = {}

    def predict(self) -> Dict[str, Any]:
        ann_doc = load_driving_annotations(
            self.annotations,
            max_frames=self.max_frames,
            source_frame="unknown",
        )
        selected_timestamps = {int(frame["timestamp"]) for frame in ann_doc["frames"]}
        ignored_objects, ignored_class_counts = _count_ignored_objects_for_timestamps(
            self.annotations,
            selected_timestamps,
        )

        frames_out: List[FrameDetections3D] = []
        gt_objects_used = 0
        false_positive_count = 0
        output_counts: Counter[str] = Counter()

        for frame_idx, frame in enumerate(ann_doc["frames"]):
            boxes: List[Detection3D] = []
            for obj in frame["objects"]:
                cls = str(obj["normalized_class"])
                keep_prob = self.vehicle_keep_prob if cls == "vehicle" else self.pedestrian_keep_prob
                if self.rng.random() > keep_prob:
                    continue
                gt_objects_used += 1
                det_id = f"det_{frame_idx:06d}_{len(boxes):03d}"
                boxes.append(self._noisy_detection(det_id, obj))
                output_counts[cls] += 1

            for fp_cls in ("vehicle", "pedestrian"):
                n_fp = self.rng.randint(0, 2 if fp_cls == "vehicle" else 1)
                for _ in range(n_fp):
                    det_id = f"det_{frame_idx:06d}_{len(boxes):03d}"
                    boxes.append(self._false_positive(det_id, fp_cls))
                    false_positive_count += 1
                    output_counts[fp_cls] += 1

            frames_out.append(
                FrameDetections3D(
                    frame_id=f"{frame_idx:06d}",
                    timestamp=int(frame["timestamp"]),
                    boxes3d=boxes,
                )
            )

        warnings = [
            COORDINATE_FRAME_NOTE,
            "Mock detections are generated from GT annotations and are for pipeline prototyping only.",
            "Yaw is normalized to [0, 2*pi).",
            "Debug GT fields, when enabled, must not be used by downstream pseudo-label generation.",
        ]
        doc = Detections3DDocument(
            source="mock_lidar_detector",
            coordinate_frame="lidar_like",
            coordinate_frame_note=COORDINATE_FRAME_NOTE,
            warnings=warnings,
            frames=frames_out,
            extra_metadata={
                "mock_config": {
                    "seed": self.seed,
                    "max_frames": self.max_frames,
                    "vehicle_keep_prob": self.vehicle_keep_prob,
                    "pedestrian_keep_prob": self.pedestrian_keep_prob,
                    "xy_std": self.xy_std,
                    "z_std": self.z_std,
                    "yaw_std": self.yaw_std,
                    "distance_adaptive_xy_noise": True,
                    "size_noise_uniform": [0.95, 1.05],
                    "true_detection_score_range": [0.45, 0.95],
                    "false_positive_score_range": [0.25, 0.55],
                },
                "summary": {
                    "input_frames": len(ann_doc["frames"]),
                    "gt_objects_considered": sum(len(frame["objects"]) for frame in ann_doc["frames"]),
                    "gt_objects_used": gt_objects_used,
                    "ignored_objects": ignored_objects,
                    "ignored_class_counts": ignored_class_counts,
                    "mock_detections_generated": sum(len(f.boxes3d) for f in frames_out),
                    "false_positives_generated": false_positive_count,
                    "per_class_output_counts": dict(output_counts),
                },
            },
        ).to_dict(include_debug_fields=self.include_debug_fields)
        self.summary = doc["metadata"]["summary"]
        return doc

    def _noisy_detection(self, det_id: str, obj: Dict[str, Any]) -> Detection3D:
        x, y, z = [float(v) for v in obj["position"]]
        length, width, height = [float(v) for v in obj["dimensions"]]
        xy_std = _distance_adaptive_xy_std(x, y, self.xy_std)
        nx = x + self.rng.gauss(0.0, xy_std)
        ny = y + self.rng.gauss(0.0, xy_std)
        nz = z + self.rng.gauss(0.0, self.z_std)
        nl = max(0.05, length * self.rng.uniform(0.95, 1.05))
        nw = max(0.05, width * self.rng.uniform(0.95, 1.05))
        nh = max(0.05, height * self.rng.uniform(0.95, 1.05))
        yaw = _wrap_yaw_0_2pi(float(obj["yaw"]) + self.rng.gauss(0.0, self.yaw_std))
        score = self.rng.uniform(0.45, 0.95)
        debug: Dict[str, Any] = {}
        if self.include_debug_fields:
            debug = {
                "mock_from_gt": True,
                "gt_id": obj.get("gt_id"),
                "original_track_id": (obj.get("raw") or {}).get("track_id"),
            }
        return Detection3D(
            det_id=det_id,
            class_name=str(obj["normalized_class"]),
            box3d_lidar=[nx, ny, nz, nl, nw, nh, yaw],
            score_3d=score,
            source="mock_lidar_detector",
            debug=debug,
        )

    def _false_positive(self, det_id: str, cls: str) -> Detection3D:
        x = self.rng.uniform(0.0, 50.0)
        y = self.rng.uniform(-25.0, 25.0)
        z = self.rng.uniform(-1.0, 2.0)
        if cls == "vehicle":
            base = (4.2, 1.8, 1.6)
        else:
            base = (0.6, 0.6, 1.7)
        dims = [max(0.05, b * self.rng.uniform(0.85, 1.15)) for b in base]
        yaw = self.rng.uniform(0.0, 2.0 * math.pi)
        score = self.rng.uniform(0.25, 0.55)
        debug: Dict[str, Any] = {}
        if self.include_debug_fields:
            debug = {"mock_from_gt": False}
        return Detection3D(
            det_id=det_id,
            class_name=cls,
            box3d_lidar=[x, y, z, dims[0], dims[1], dims[2], yaw],
            score_3d=score,
            source="mock_lidar_detector",
            debug=debug,
        )


def _count_ignored_objects_for_timestamps(annotations: Path, selected_timestamps: set[int]) -> Tuple[int, Dict[str, int]]:
    with annotations.open("r", encoding="utf-8") as f:
        raw_doc = json.load(f)
    ignored = 0
    counts: Counter[str] = Counter()
    for track in raw_doc.get("tracks", []):
        if not isinstance(track, dict):
            continue
        raw_class = str(track.get("object_type", ""))
        if normalize_class(raw_class) is not None:
            continue
        for ts in track.get("timestamps", []):
            try:
                ts_int = int(ts)
            except (TypeError, ValueError):
                continue
            if ts_int in selected_timestamps:
                ignored += 1
                counts[raw_class] += 1
    return ignored, dict(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock LiDAR detector outputs from DrivIng annotations")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--vehicle-keep-prob", type=float, default=0.85)
    parser.add_argument("--pedestrian-keep-prob", type=float, default=0.65)
    parser.add_argument("--xy-std", type=float, default=0.30)
    parser.add_argument("--z-std", type=float, default=0.10)
    parser.add_argument("--yaw-std", type=float, default=0.08)
    parser.add_argument("--include-debug-fields", action="store_true")
    return parser.parse_args()


def _print_summary(doc: Dict[str, Any], output: Path) -> None:
    metadata = doc.get("metadata", {})
    summary = metadata.get("summary", {})
    print("Mock LiDAR detector summary")
    print(f"- input frames: {summary.get('input_frames', 0)}")
    print(f"- GT objects considered: {summary.get('gt_objects_considered', 0)}")
    print(f"- GT objects used: {summary.get('gt_objects_used', 0)}")
    print(f"- ignored objects: {summary.get('ignored_objects', 0)}")
    print(f"- ignored class counts: {summary.get('ignored_class_counts', {})}")
    print(f"- mock detections generated: {summary.get('mock_detections_generated', 0)}")
    print(f"- false positives generated: {summary.get('false_positives_generated', 0)}")
    print(f"- per-class output counts: {summary.get('per_class_output_counts', {})}")
    print(f"- coordinate frame: {metadata.get('coordinate_frame')}")
    print(f"- coordinate frame note: {metadata.get('coordinate_frame_note')}")
    for warning in metadata.get("warnings", []):
        print(f"WARNING: {warning}")
    errors = validate_detections_json(doc)
    print(f"- validation errors: {len(errors)}")
    print(f"Wrote detections: {output}")
    preview_frame = None
    for frame in doc.get("frames", []):
        boxes = frame.get("boxes3d") or []
        if boxes:
            preview_frame = {
                "frame_id": frame.get("frame_id"),
                "timestamp": frame.get("timestamp"),
                "boxes3d": boxes[:2],
            }
            break
    if preview_frame:
        print("Preview canonical frame:")
        print(json.dumps(preview_frame, ensure_ascii=False, indent=2)[:1600])


def main() -> int:
    args = parse_args()
    provider = MockLidarDetectorProvider(
        annotations=args.annotations,
        seed=args.seed,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        vehicle_keep_prob=args.vehicle_keep_prob,
        pedestrian_keep_prob=args.pedestrian_keep_prob,
        xy_std=args.xy_std,
        z_std=args.z_std,
        yaw_std=args.yaw_std,
        include_debug_fields=args.include_debug_fields,
    )
    doc = provider.predict()
    save_detections_json(doc, args.output)
    # Round-trip once so the CLI proves the saved file validates, not just the in-memory dict.
    loaded = load_detections_json(args.output)
    _print_summary(loaded, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
