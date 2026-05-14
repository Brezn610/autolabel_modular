from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .annotation_projection_probe import (
    _corners_from_center_lwh_yaw,
    _load_calibration,
    _load_timesync,
    _resolve_calibration_path,
)
from .driving_annotations import load_driving_annotations


COLORS = {
    "lidar_assumed": (0, 180, 255),
    "vehicle_assumed": (80, 220, 80),
}


def _load_lidar_xyz(data_root: Path, lidar_filename: str) -> np.ndarray:
    path = data_root / "middle_lidar" / lidar_filename
    if not path.is_file():
        raise FileNotFoundError(f"LiDAR file not found: {path}")
    npz = np.load(path)
    if not all(k in npz.files for k in ("x", "y", "z")):
        raise ValueError(f"LiDAR npz must contain x,y,z arrays: {path}")
    return np.stack([npz["x"], npz["y"], npz["z"]], axis=1).astype(np.float64)


def _transform_points(points: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)])
    return (T_dst_src @ pts_h.T).T[:, :3]


def _polygon_contains_points(points_xy: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0,), dtype=bool)
    # poly corners are ordered around the rectangle. Check same-side cross products.
    signs = []
    for i in range(poly_xy.shape[0]):
        a = poly_xy[i]
        b = poly_xy[(i + 1) % poly_xy.shape[0]]
        edge = b - a
        rel = points_xy - a.reshape(1, 2)
        signs.append(edge[0] * rel[:, 1] - edge[1] * rel[:, 0])
    cross = np.stack(signs, axis=1)
    return np.all(cross >= -1e-9, axis=1) | np.all(cross <= 1e-9, axis=1)


def _count_points_in_bev_box(points_xy: np.ndarray, box_xy: np.ndarray) -> int:
    lo = box_xy.min(axis=0)
    hi = box_xy.max(axis=0)
    candidate = (
        (points_xy[:, 0] >= lo[0])
        & (points_xy[:, 0] <= hi[0])
        & (points_xy[:, 1] >= lo[1])
        & (points_xy[:, 1] <= hi[1])
    )
    if not np.any(candidate):
        return 0
    return int(np.count_nonzero(_polygon_contains_points(points_xy[candidate], box_xy)))


def _world_to_px(
    xy: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    image_size: int,
) -> np.ndarray:
    x_min, x_max = x_range
    y_min, y_max = y_range
    sx = (image_size - 1) / max(1e-6, x_max - x_min)
    sy = (image_size - 1) / max(1e-6, y_max - y_min)
    u = (xy[:, 1] - y_min) * sy
    v = (x_max - xy[:, 0]) * sx
    return np.stack([u, v], axis=1)


def _draw_label(img: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_bev(
    points_lidar: np.ndarray,
    boxes_by_mode: Dict[str, List[Dict[str, Any]]],
    out_path: Path,
    *,
    image_size: int,
    max_points_draw: int,
) -> None:
    canvas = np.full((image_size, image_size, 3), 20, dtype=np.uint8)
    xy = points_lidar[:, :2]
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if xy.shape[0] > max_points_draw:
        rng = np.random.default_rng(0)
        xy_draw = xy[rng.choice(xy.shape[0], size=max_points_draw, replace=False)]
    else:
        xy_draw = xy

    all_xy = [xy_draw]
    for entries in boxes_by_mode.values():
        for entry in entries:
            all_xy.append(entry["corners_lidar"][:4, :2])
    xy_for_range = np.vstack(all_xy) if all_xy else xy_draw
    pad = 5.0
    x_range = (float(np.percentile(xy_for_range[:, 0], 1)) - pad, float(np.percentile(xy_for_range[:, 0], 99)) + pad)
    y_range = (float(np.percentile(xy_for_range[:, 1], 1)) - pad, float(np.percentile(xy_for_range[:, 1], 99)) + pad)

    pts_px = _world_to_px(xy_draw, x_range, y_range, image_size)
    for p in pts_px.astype(np.int32):
        u, v = int(p[0]), int(p[1])
        if 0 <= u < image_size and 0 <= v < image_size:
            canvas[v, u] = (90, 90, 90)

    for mode, entries in boxes_by_mode.items():
        color = COLORS[mode]
        for entry in entries:
            rect_xy = entry["corners_lidar"][:4, :2]
            rect_px = _world_to_px(rect_xy, x_range, y_range, image_size).astype(np.int32)
            cv2.polylines(canvas, [rect_px.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            label_px = rect_px[0]
            _draw_label(
                canvas,
                f"{mode}:{entry['class']}:{entry['track_id']} p={entry['points_in_box']}",
                (int(label_px[0]), int(label_px[1])),
                color,
            )

    cv2.putText(canvas, "BEV: x forward, y lateral", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    cv2.imwrite(str(out_path), canvas)


def _support_threshold(normalized_class: str) -> int:
    if normalized_class == "pedestrian":
        return 2
    return 5


def _summarize_point_counts(point_counts: List[int]) -> Dict[str, float]:
    if not point_counts:
        return {"average_points_per_box": 0.0, "median_points_per_box": 0.0}
    arr = np.asarray(point_counts, dtype=np.float64)
    return {
        "average_points_per_box": float(np.mean(arr)),
        "median_points_per_box": float(np.median(arr)),
    }


def inspect_ego_world_info(data_root: Path, calibration: Dict[str, Any]) -> Dict[str, Any]:
    vehicle_dir = data_root / "vehicle_state"
    sweeps_vehicle_dir = data_root / "sweeps" / "vehicle_state"
    sample: Dict[str, Any] = {}
    sample_path = None
    if vehicle_dir.is_dir():
        files = sorted(vehicle_dir.glob("*.json"))
        if files:
            sample_path = files[0]
    if sample_path is not None:
        try:
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            sample = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "vehicle_state_dir_exists": vehicle_dir.is_dir(),
        "vehicle_state_file_count": len(list(vehicle_dir.glob("*.json"))) if vehicle_dir.is_dir() else 0,
        "sweeps_vehicle_state_dir_exists": sweeps_vehicle_dir.is_dir(),
        "sweeps_vehicle_state_file_count": len(list(sweeps_vehicle_dir.glob("*.json"))) if sweeps_vehicle_dir.is_dir() else 0,
        "sample_vehicle_state_path": str(sample_path) if sample_path is not None else "",
        "sample_vehicle_state_fields": list(sample.keys()),
        "sample_vehicle_state": sample,
        "calibration_middle_lidar_extrinsics": list((calibration.get("middle_lidar") or {}).get("extrinsics", {}).keys()),
        "calibration_state_keys": list((calibration.get("state") or {}).keys()) if isinstance(calibration.get("state"), dict) else [],
        "assessment": (
            "vehicle_state has pos_rel_x, pos_rel_y, height_msl, roll, pitch, yaw fields and calibration has middle_lidar.extrinsics.vTl. "
            "This likely supports ego/world transforms, but annotation position frame and transform direction still need explicit validation before use."
        ),
    }


def _recommend(bev_totals: Dict[str, Any]) -> str:
    def score(mode: str) -> float:
        s = bev_totals[mode]
        return float(s["percentage_with_lidar_support"]) + 0.25 * float(s["percentage_center_in_lidar_range"])

    lidar_score = score("lidar_assumed")
    vehicle_score = score("vehicle_assumed")
    if max(lidar_score, vehicle_score) < 0.20:
        return "annotations likely world-frame; need ego/world transform"
    if lidar_score > vehicle_score + 0.10:
        return "annotations appear LiDAR-like"
    if vehicle_score > lidar_score + 0.10:
        return "annotations appear vehicle/ego-like"
    return "inconclusive; manual inspection required"


def run_bev_probe(
    *,
    annotations: Path,
    calibration: Path,
    data_root: Path,
    output: Path,
    max_frames: int,
    image_size: int,
    max_points_draw: int,
) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    calib_path = _resolve_calibration_path(calibration, data_root)
    calib = _load_calibration(calib_path)
    timesync = _load_timesync(data_root)
    ann_doc = load_driving_annotations(annotations, max_frames=max_frames, source_frame="unknown")

    vTl = np.asarray(calib["middle_lidar"]["extrinsics"]["vTl"], dtype=np.float64)
    T_lidar_vehicle = np.linalg.inv(vTl)
    totals: Dict[str, Dict[str, Any]] = {
        mode: {
            "boxes_drawn": 0,
            "centers_in_lidar_range": 0,
            "boxes_with_lidar_support": 0,
            "point_counts": [],
        }
        for mode in COLORS
    }
    frame_summaries: List[Dict[str, Any]] = []
    missing_lidar: List[Dict[str, Any]] = []

    for frame in ann_doc["frames"]:
        ts = int(frame["timestamp"])
        files = timesync.get(ts)
        if files is None or not files.get("middle_lidar"):
            missing_lidar.append({"timestamp": ts, "reason": "middle_lidar_missing_in_timesync"})
            continue
        lidar_name = files["middle_lidar"]
        try:
            points = _load_lidar_xyz(data_root, lidar_name)
        except Exception as exc:  # noqa: BLE001
            missing_lidar.append({"timestamp": ts, "lidar_file": lidar_name, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        points_xy = points[:, :2]
        finite = np.isfinite(points_xy).all(axis=1)
        points_xy = points_xy[finite]
        lidar_min = points_xy.min(axis=0)
        lidar_max = points_xy.max(axis=0)
        boxes_by_mode: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in COLORS}
        frame_summary = {
            "timestamp": ts,
            "lidar_file": str(data_root / "middle_lidar" / lidar_name),
            "num_gt_objects": len(frame["objects"]),
            "assumptions": {},
        }

        for mode in COLORS:
            mode_counts: List[int] = []
            centers_in_range = 0
            supported = 0
            for obj in frame["objects"]:
                corners = _corners_from_center_lwh_yaw(obj["position"], obj["dimensions"], float(obj["yaw"]))
                if mode == "vehicle_assumed":
                    corners_lidar = _transform_points(corners, T_lidar_vehicle)
                    center_lidar = _transform_points(np.asarray([obj["position"]], dtype=np.float64), T_lidar_vehicle)[0]
                else:
                    corners_lidar = corners
                    center_lidar = np.asarray(obj["position"], dtype=np.float64)

                rect_xy = corners_lidar[:4, :2]
                n_inside = _count_points_in_bev_box(points_xy, rect_xy)
                threshold = _support_threshold(str(obj["normalized_class"]))
                has_support = n_inside >= threshold
                center_ok = bool(
                    lidar_min[0] <= center_lidar[0] <= lidar_max[0]
                    and lidar_min[1] <= center_lidar[1] <= lidar_max[1]
                )
                if center_ok:
                    centers_in_range += 1
                if has_support:
                    supported += 1
                mode_counts.append(n_inside)
                boxes_by_mode[mode].append(
                    {
                        "corners_lidar": corners_lidar,
                        "class": obj["normalized_class"],
                        "track_id": obj["gt_id"],
                        "points_in_box": int(n_inside),
                    }
                )

            boxes_drawn = len(frame["objects"])
            totals[mode]["boxes_drawn"] += boxes_drawn
            totals[mode]["centers_in_lidar_range"] += centers_in_range
            totals[mode]["boxes_with_lidar_support"] += supported
            totals[mode]["point_counts"].extend(mode_counts)
            frame_summary["assumptions"][mode] = {
                "boxes_drawn": int(boxes_drawn),
                "centers_in_lidar_range": int(centers_in_range),
                "boxes_with_lidar_support": int(supported),
                "percentage_with_lidar_support": float(supported / max(1, boxes_drawn)),
                **_summarize_point_counts(mode_counts),
            }

        out_path = output / f"bev_frame_{ts}.png"
        _draw_bev(points, boxes_by_mode, out_path, image_size=image_size, max_points_draw=max_points_draw)
        frame_summary["debug_image"] = str(out_path)
        frame_summaries.append(frame_summary)

    totals_plain: Dict[str, Any] = {}
    for mode, stats in totals.items():
        boxes = max(1, int(stats["boxes_drawn"]))
        totals_plain[mode] = {
            "boxes_drawn": int(stats["boxes_drawn"]),
            "centers_in_lidar_range": int(stats["centers_in_lidar_range"]),
            "percentage_center_in_lidar_range": float(stats["centers_in_lidar_range"] / boxes),
            "boxes_with_lidar_support": int(stats["boxes_with_lidar_support"]),
            "percentage_with_lidar_support": float(stats["boxes_with_lidar_support"] / boxes),
            **_summarize_point_counts([int(x) for x in stats["point_counts"]]),
        }

    summary = {
        "inputs": {
            "annotations": str(annotations),
            "calibration": str(calib_path),
            "data_root": str(data_root),
            "output": str(output),
            "max_frames": int(max_frames),
            "image_size": int(image_size),
        },
        "warnings": [
            "BEV support is a rough alignment diagnostic only; it does not create 3D candidates.",
            "vehicle_assumed boxes are transformed to LiDAR frame using inverse middle_lidar.extrinsics.vTl before BEV comparison.",
            "world_assumed is not evaluated because the annotation coordinate frame is not declared.",
        ],
        "bev_totals": totals_plain,
        "frames": frame_summaries,
        "missing_lidar": missing_lidar,
        "ego_world_info": inspect_ego_world_info(data_root, calib),
    }
    summary["recommendation"] = _recommend(totals_plain)
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw annotation boxes over LiDAR BEV under coordinate-frame assumptions")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/debug/annotation_bev_probe"))
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=1000)
    parser.add_argument("--max-points-draw", type=int, default=120_000)
    return parser.parse_args()


def _print_summary(summary: Dict[str, Any]) -> None:
    print("Annotation BEV probe")
    print(f"- output: {summary['inputs']['output']}")
    for mode, stats in summary["bev_totals"].items():
        print(
            f"- {mode}: boxes={stats['boxes_drawn']} "
            f"centers_in_range={stats['centers_in_lidar_range']} "
            f"center_pct={100.0 * stats['percentage_center_in_lidar_range']:.2f}% "
            f"supported={stats['boxes_with_lidar_support']} "
            f"support_pct={100.0 * stats['percentage_with_lidar_support']:.2f}% "
            f"avg_points={stats['average_points_per_box']:.2f} "
            f"median_points={stats['median_points_per_box']:.2f}"
        )
    e = summary["ego_world_info"]
    print(
        f"- ego/world files: vehicle_state={e['vehicle_state_file_count']} "
        f"sweeps_vehicle_state={e['sweeps_vehicle_state_file_count']}"
    )
    print(f"- recommendation: {summary['recommendation']}")
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")


def main() -> int:
    args = parse_args()
    summary = run_bev_probe(
        annotations=args.annotations,
        calibration=args.calibration,
        data_root=args.data_root,
        output=args.output,
        max_frames=args.max_frames,
        image_size=args.image_size,
        max_points_draw=args.max_points_draw,
    )
    _print_summary(summary)
    print(f"Wrote summary: {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
