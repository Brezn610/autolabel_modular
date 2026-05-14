from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


VEHICLE_CLASSES = {"car", "truck", "bus", "trailer", "van"}
PEDESTRIAN_CLASSES = {"pedestrian"}


@dataclass
class AnnotationSummary:
    num_tracks: int = 0
    num_frame_entries: int = 0
    normalized_class_counts: Counter[str] = field(default_factory=Counter)
    raw_class_track_counts: Counter[str] = field(default_factory=Counter)
    ignored_object_entries: int = 0
    ignored_track_counts: Counter[str] = field(default_factory=Counter)
    timestamp_min: Optional[int] = None
    timestamp_max: Optional[int] = None
    position_min: Optional[List[float]] = None
    position_max: Optional[List[float]] = None
    dimensions_min: Optional[List[float]] = None
    dimensions_max: Optional[List[float]] = None
    yaw_min: Optional[float] = None
    yaw_max: Optional[float] = None
    malformed_tracks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_tracks": self.num_tracks,
            "num_frame_entries": self.num_frame_entries,
            "normalized_class_counts": dict(self.normalized_class_counts),
            "raw_class_track_counts": dict(self.raw_class_track_counts),
            "ignored_object_entries": self.ignored_object_entries,
            "ignored_track_counts": dict(self.ignored_track_counts),
            "timestamp_range": [self.timestamp_min, self.timestamp_max],
            "position_min": self.position_min,
            "position_max": self.position_max,
            "dimensions_min": self.dimensions_min,
            "dimensions_max": self.dimensions_max,
            "yaw_range": [self.yaw_min, self.yaw_max],
            "malformed_tracks": self.malformed_tracks,
        }


def normalize_class(raw_class: Any) -> Optional[str]:
    key = str(raw_class or "").strip().lower()
    if key in VEHICLE_CLASSES:
        return "vehicle"
    if key in PEDESTRIAN_CLASSES:
        return "pedestrian"
    return None


def _as_float_triplet(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _update_vec_range(
    current_min: Optional[List[float]],
    current_max: Optional[List[float]],
    value: List[float],
) -> Tuple[List[float], List[float]]:
    if current_min is None or current_max is None:
        return list(value), list(value)
    return (
        [min(a, b) for a, b in zip(current_min, value)],
        [max(a, b) for a, b in zip(current_max, value)],
    )


def _update_scalar_range(
    current_min: Optional[float],
    current_max: Optional[float],
    value: float,
) -> Tuple[float, float]:
    if current_min is None or current_max is None:
        return value, value
    return min(current_min, value), max(current_max, value)


def _update_int_range(
    current_min: Optional[int],
    current_max: Optional[int],
    value: int,
) -> Tuple[int, int]:
    if current_min is None or current_max is None:
        return value, value
    return min(current_min, value), max(current_max, value)


def _extract_dimensions(track: Dict[str, Any]) -> Optional[List[float]]:
    dims = track.get("dimensions")
    if not isinstance(dims, list) or not dims:
        return None
    # DrivIng stores dimensions once per track in the inspected file:
    # "dimensions": [[length, width, height]]
    first = dims[0]
    return _as_float_triplet(first)


def _iter_track_entries(track: Dict[str, Any]) -> Iterable[Tuple[int, List[float], float]]:
    timestamps = track.get("timestamps")
    positions = track.get("positions")
    orientations = track.get("orientations")
    if not isinstance(timestamps, list) or not isinstance(positions, list) or not isinstance(orientations, list):
        return
    n = min(len(timestamps), len(positions), len(orientations))
    for i in range(n):
        try:
            ts = int(timestamps[i])
            pos = _as_float_triplet(positions[i])
            yaw = float(orientations[i])
        except (TypeError, ValueError):
            continue
        if pos is None:
            continue
        yield ts, pos, yaw


def load_driving_annotations(
    annotations_path: Path,
    *,
    max_frames: Optional[int] = None,
    source_frame: str = "unknown",
) -> Dict[str, Any]:
    """Load DrivIng track-major annotations into a frame-major preview structure.

    The source coordinate frame is deliberately not inferred here. The inspected
    file stores numeric positions and yaw-like orientations, but it does not by
    itself state whether those positions are LiDAR, ego/vehicle, or world-frame.
    """
    with annotations_path.open("r", encoding="utf-8") as f:
        raw_doc = json.load(f)

    tracks = raw_doc.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("annotations JSON must contain a top-level list field: tracks")

    summary = AnnotationSummary(num_tracks=len(tracks))
    warnings = [
        "Coordinate frame is not declared in annotations.json; source_frame is set to 'unknown'.",
        "Orientation is treated as a yaw-like scalar for normalization only; axis convention is unverified.",
        "Dimensions are assumed to be [length, width, height] because each inspected track stores dimensions as [[L, W, H]].",
    ]
    assumptions = [
        "Top-level annotations are track-major: each track has parallel timestamps, positions, and orientations arrays.",
        "Only raw classes car, truck, bus, trailer, van, and pedestrian are emitted in normalized objects.",
    ]

    by_timestamp: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    all_timestamps: set[int] = set()

    for track in tracks:
        if not isinstance(track, dict):
            summary.malformed_tracks += 1
            continue

        raw_class = str(track.get("object_type", ""))
        normalized_class = normalize_class(raw_class)
        summary.raw_class_track_counts[raw_class] += 1

        dimensions = _extract_dimensions(track)
        if dimensions is not None:
            summary.dimensions_min, summary.dimensions_max = _update_vec_range(
                summary.dimensions_min,
                summary.dimensions_max,
                dimensions,
            )

        entries = list(_iter_track_entries(track))
        if not entries or dimensions is None:
            summary.malformed_tracks += 1
            continue

        if normalized_class is None:
            summary.ignored_track_counts[raw_class] += 1
            summary.ignored_object_entries += len(entries)
            for ts, pos, yaw in entries:
                all_timestamps.add(ts)
                summary.timestamp_min, summary.timestamp_max = _update_int_range(
                    summary.timestamp_min,
                    summary.timestamp_max,
                    ts,
                )
                summary.position_min, summary.position_max = _update_vec_range(
                    summary.position_min,
                    summary.position_max,
                    pos,
                )
                summary.yaw_min, summary.yaw_max = _update_scalar_range(summary.yaw_min, summary.yaw_max, yaw)
            continue

        for ts, pos, yaw in entries:
            all_timestamps.add(ts)
            summary.timestamp_min, summary.timestamp_max = _update_int_range(
                summary.timestamp_min,
                summary.timestamp_max,
                ts,
            )
            summary.position_min, summary.position_max = _update_vec_range(
                summary.position_min,
                summary.position_max,
                pos,
            )
            summary.yaw_min, summary.yaw_max = _update_scalar_range(summary.yaw_min, summary.yaw_max, yaw)
            summary.normalized_class_counts[normalized_class] += 1

            by_timestamp[ts].append(
                {
                    "gt_id": str(track.get("track_id")),
                    "raw_class": raw_class,
                    "normalized_class": normalized_class,
                    "position": pos,
                    "dimensions": dimensions,
                    "yaw": yaw,
                    "source_frame": source_frame,
                    "raw": {
                        "track_id": track.get("track_id"),
                        "object_type": raw_class,
                        "attributes": track.get("attributes", {}),
                        "timestamp": ts,
                        "position": pos,
                        "orientation": yaw,
                        "dimensions": dimensions,
                    },
                }
            )

    sorted_timestamps = sorted(by_timestamp)
    if max_frames is not None:
        sorted_timestamps = sorted_timestamps[: max(0, int(max_frames))]

    frames = [
        {
            "frame_id": str(ts),
            "timestamp": ts,
            "objects": by_timestamp[ts],
        }
        for ts in sorted_timestamps
    ]

    summary.num_frame_entries = len(all_timestamps)
    metadata = {
        "source_path": str(annotations_path),
        "source_id": raw_doc.get("id"),
        "source_timestamp": raw_doc.get("timestamp"),
        "source_frame": source_frame,
        "assumptions": assumptions,
        "warnings": warnings,
        "summary": summary.to_dict(),
    }
    return {"frames": frames, "metadata": metadata}


def _format_summary(summary: Dict[str, Any], preview_frames: int) -> str:
    lines = [
        "DrivIng annotations summary",
        f"- tracks: {summary['num_tracks']}",
        f"- unique timestamp/frame entries: {summary['num_frame_entries']}",
        f"- preview frames written: {preview_frames}",
        f"- normalized object entries: {summary['normalized_class_counts']}",
        f"- ignored object entries: {summary['ignored_object_entries']}",
        f"- ignored track counts: {summary['ignored_track_counts']}",
        f"- timestamp range: {summary['timestamp_range']}",
        f"- position min/max: {summary['position_min']} / {summary['position_max']}",
        f"- dimensions min/max: {summary['dimensions_min']} / {summary['dimensions_max']}",
        f"- yaw/orientation range: {summary['yaw_range']}",
        f"- malformed tracks: {summary['malformed_tracks']}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and normalize DrivIng annotations.json")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    doc = load_driving_annotations(args.annotations, max_frames=args.max_frames, source_frame="unknown")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    summary = doc["metadata"]["summary"]
    print(_format_summary(summary, preview_frames=len(doc["frames"])))
    for warning in doc["metadata"]["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Wrote preview: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
