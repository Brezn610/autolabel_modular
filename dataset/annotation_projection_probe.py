from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .driving_annotations import load_driving_annotations


DEFAULT_CAMERAS = [
    "front_left_camera",
    "front_right_camera",
    "left_camera",
    "right_camera",
    "back_left_camera",
    "back_right_camera",
]

COLORS = {
    "lidar_assumed": (0, 180, 255),
    "vehicle_assumed": (80, 220, 80),
}

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _load_calibration(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_intrinsic_to_K(intrinsics: Dict[str, Any]) -> np.ndarray:
    m = np.asarray(intrinsics["IntrinsicMatrix"], dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"IntrinsicMatrix must be 3x3, got {m.shape}")
    fx = m[0, 0]
    skew = m[1, 0]
    fy = m[1, 1]
    cx = m[2, 0]
    cy = m[2, 1]
    return np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _distortion_coeffs(intrinsics: Dict[str, Any]) -> np.ndarray:
    if "DistortionCoefficients" in intrinsics:
        return np.asarray(intrinsics["DistortionCoefficients"], dtype=np.float64).reshape(-1)
    radial = [float(x) for x in intrinsics.get("RadialDistortion", [])]
    tangential = [float(x) for x in intrinsics.get("TangentialDistortion", [])]
    if len(radial) < 2:
        return np.asarray([], dtype=np.float64)
    coeffs = [
        radial[0],
        radial[1],
        tangential[0] if tangential else 0.0,
        tangential[1] if len(tangential) > 1 else 0.0,
    ]
    if len(radial) >= 3:
        coeffs.append(radial[2])
    return np.asarray(coeffs, dtype=np.float64)


def _image_size_hw(intrinsics: Dict[str, Any]) -> Tuple[int, int]:
    h, w = intrinsics["ImageSize"]
    return int(h), int(w)


def _load_timesync(data_root: Path) -> Dict[int, Dict[str, str]]:
    csv_path = data_root / "timesync_info.csv"
    xlsx_path = data_root / "timesync_info.xlsx"
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.reader(f))
    elif xlsx_path.is_file():
        try:
            import pandas as pd
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("timesync_info.xlsx requires pandas/openpyxl") from exc
        rows = pd.read_excel(xlsx_path, header=None, engine="openpyxl").astype(str).values.tolist()
    else:
        raise FileNotFoundError(f"Could not find {csv_path} or {xlsx_path}")

    sensor_map: Dict[str, List[str]] = {}
    for row in rows[1:]:
        if row and row[0]:
            sensor_map[str(row[0]).strip()] = [str(x).strip() for x in row[1:]]
    if "timestamp_nanoseconds" not in sensor_map:
        raise ValueError("timesync table is missing timestamp_nanoseconds row")

    n = min(len(v) for v in sensor_map.values())
    out: Dict[int, Dict[str, str]] = {}
    for i in range(n):
        try:
            ts = int(sensor_map["timestamp_nanoseconds"][i])
        except ValueError:
            continue
        out[ts] = {k: vals[i] for k, vals in sensor_map.items() if k != "timestamp_nanoseconds"}
    return out


def _corners_from_center_lwh_yaw(
    center: Iterable[float],
    dimensions_lwh: Iterable[float],
    yaw: float,
) -> np.ndarray:
    cx, cy, cz = [float(v) for v in center]
    length, width, height = [float(v) for v in dimensions_lwh]
    hx, hy, hz = 0.5 * length, 0.5 * width, 0.5 * height
    local = np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    cyaw, syaw = math.cos(float(yaw)), math.sin(float(yaw))
    r = np.array([[cyaw, -syaw, 0.0], [syaw, cyaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (r @ local.T).T + np.array([[cx, cy, cz]], dtype=np.float64)


def _project_points(points: np.ndarray, T_dst_src: np.ndarray, K: np.ndarray, dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts_h = np.hstack([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)])
    pts_cam = (T_dst_src @ pts_h.T).T[:, :3]
    img_pts, _ = cv2.projectPoints(pts_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return np.asarray(img_pts.reshape(-1, 2), dtype=np.float64), np.asarray(pts_cam[:, 2], dtype=np.float64)


def _draw_box(
    image: np.ndarray,
    uv: np.ndarray,
    z_cam: np.ndarray,
    label: str,
    color: Tuple[int, int, int],
) -> None:
    h, w = image.shape[:2]
    for i, j in BOX_EDGES:
        if z_cam[i] <= 0.05 or z_cam[j] <= 0.05:
            continue
        p1 = (int(round(float(uv[i, 0]))), int(round(float(uv[i, 1]))))
        p2 = (int(round(float(uv[j, 0]))), int(round(float(uv[j, 1]))))
        ok, a, b = cv2.clipLine((0, 0, w, h), p1, p2)
        if ok:
            cv2.line(image, a, b, color, 2, cv2.LINE_AA)

    front = z_cam > 0.05
    if not np.any(front):
        return
    valid_uv = uv[front]
    finite = np.isfinite(valid_uv).all(axis=1)
    if not np.any(finite):
        return
    x = int(np.clip(np.nanmin(valid_uv[finite, 0]), 0, max(0, w - 1)))
    y = int(np.clip(np.nanmin(valid_uv[finite, 1]), 0, max(0, h - 1)))
    cv2.putText(image, label, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _projection_stats(uv: np.ndarray, z_cam: np.ndarray, image_wh: Tuple[int, int]) -> Dict[str, Any]:
    w, h = image_wh
    front = z_cam > 0.05
    any_front = bool(np.any(front))
    if not any_front:
        return {
            "any_front": False,
            "inside_image": False,
            "front_corners": 0,
            "bbox_xyxy": None,
            "bbox_area_ratio": 0.0,
            "reasonable_projected_box": False,
            "suspicious": ["all_corners_behind_camera"],
        }

    pts = uv[front]
    finite = np.isfinite(pts).all(axis=1)
    if not np.any(finite):
        return {
            "any_front": True,
            "inside_image": False,
            "front_corners": int(np.count_nonzero(front)),
            "bbox_xyxy": None,
            "bbox_area_ratio": 0.0,
            "reasonable_projected_box": False,
            "suspicious": ["non_finite_projection"],
        }

    pts = pts[finite]
    x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    intersects = x2 >= 0 and y2 >= 0 and x1 < w and y1 < h
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    area_ratio = (bw * bh) / max(1.0, float(w * h))
    suspicious: List[str] = []
    positive_size = bw > 1.0 and bh > 1.0
    center_near = (-0.25 * w) <= cx <= (1.25 * w) and (-0.25 * h) <= cy <= (1.25 * h)
    absurdly_large = area_ratio > 0.8 or bw > 2.0 * w or bh > 2.0 * h
    if int(np.count_nonzero(front)) < 4:
        suspicious.append("mostly_behind_camera")
    if absurdly_large:
        suspicious.append("extremely_large_projected_box")
    if not intersects:
        suspicious.append("outside_image")
    if not positive_size:
        suspicious.append("non_positive_projected_box_size")
    if not center_near:
        suspicious.append("projected_box_center_far_outside_image")
    reasonable = bool(intersects and positive_size and center_near and not absurdly_large)
    return {
        "any_front": True,
        "inside_image": bool(intersects),
        "front_corners": int(np.count_nonzero(front)),
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_width": float(bw),
        "bbox_height": float(bh),
        "bbox_center": [float(cx), float(cy)],
        "bbox_area_ratio": float(area_ratio),
        "reasonable_projected_box": reasonable,
        "suspicious": suspicious,
    }


def _resolve_calibration_path(calibration: Path, data_root: Path) -> Path:
    if calibration.is_file():
        return calibration
    fallback = data_root / calibration
    if fallback.is_file():
        return fallback
    if calibration.name == "calibration.json" and (data_root / "calibration.json").is_file():
        return data_root / "calibration.json"
    raise FileNotFoundError(f"Could not find calibration file: {calibration}")


def _camera_image_path(data_root: Path, camera: str, filename: str) -> Path:
    p = data_root / camera / filename
    if p.is_file():
        return p
    direct = data_root / filename
    if direct.is_file():
        return direct
    return p


def _score_mode(mode_summary: Dict[str, Any]) -> float:
    if "any_camera_visible_pct" in mode_summary:
        visible_rate = float(mode_summary["any_camera_visible_pct"])
        suspicious_rate = float(mode_summary["objects_only_suspicious"]) / max(1, int(mode_summary["total_objects_tested"]))
        return visible_rate - 0.5 * suspicious_rate
    projected = max(1, int(mode_summary["boxes_projected"]))
    inside_rate = float(mode_summary["boxes_inside_image"]) / projected
    suspicious_rate = float(mode_summary["suspicious_cases"]) / projected
    return inside_rate - 0.5 * suspicious_rate


def _recommend(summary: Dict[str, Any]) -> str:
    modes = summary.get("object_visibility_totals") or summary["assumption_totals"]
    lidar_score = _score_mode(modes["lidar_assumed"])
    vehicle_score = _score_mode(modes["vehicle_assumed"])
    if max(lidar_score, vehicle_score) < 0.15:
        return "neither assumption is reliable; likely needs ego/world transform"
    if lidar_score > vehicle_score + 0.1:
        return "lidar_assumed appears plausible"
    if vehicle_score > lidar_score + 0.1:
        return "vehicle_assumed appears plausible"
    return "both assumptions are inconclusive; inspect debug images before choosing a frame"


def run_probe(
    *,
    annotations: Path,
    calibration: Path,
    data_root: Path,
    output: Path,
    max_frames: int,
    camera_name: Optional[str],
) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    calib_path = _resolve_calibration_path(calibration, data_root)
    calib = _load_calibration(calib_path)
    timesync = _load_timesync(data_root)
    ann_doc = load_driving_annotations(annotations, max_frames=max_frames, source_frame="unknown")
    cameras = [camera_name] if camera_name else [c for c in DEFAULT_CAMERAS if c in calib]

    vTl = np.asarray(calib["middle_lidar"]["extrinsics"]["vTl"], dtype=np.float64)
    frame_summaries: List[Dict[str, Any]] = []
    object_visibility: Dict[str, Dict[str, Dict[str, Any]]] = {mode: {} for mode in COLORS}
    totals: Dict[str, Dict[str, Any]] = {
        mode: {
            "boxes_projected": 0,
            "boxes_any_front": 0,
            "boxes_inside_image": 0,
            "suspicious_cases": 0,
            "suspicious_reasons": defaultdict(int),
        }
        for mode in COLORS
    }
    missing_images: List[Dict[str, Any]] = []

    for frame in ann_doc["frames"]:
        ts = int(frame["timestamp"])
        for obj in frame["objects"]:
            key = f"{ts}:{obj['gt_id']}"
            for mode in COLORS:
                object_visibility[mode].setdefault(
                    key,
                    {
                        "frame_id": str(ts),
                        "timestamp": ts,
                        "track_id": obj["gt_id"],
                        "class": obj["normalized_class"],
                        "position": obj["position"],
                        "dimensions": obj["dimensions"],
                        "yaw": obj["yaw"],
                        "cameras_tested": 0,
                        "visible_cameras": 0,
                        "any_front_cameras": 0,
                        "suspicious_projection_cameras": 0,
                        "suspicious_reasons": defaultdict(int),
                    },
                )
        files = timesync.get(ts)
        if files is None:
            frame_summaries.append({"timestamp": ts, "warning": "timestamp_not_found_in_timesync"})
            continue

        for camera in cameras:
            image_name = files.get(camera)
            if not image_name:
                missing_images.append({"timestamp": ts, "camera": camera, "reason": "camera_missing_in_timesync"})
                continue
            image_path = _camera_image_path(data_root, camera, image_name)
            image = cv2.imread(str(image_path))
            if image is None:
                missing_images.append({"timestamp": ts, "camera": camera, "path": str(image_path), "reason": "image_not_readable"})
                continue

            intr = calib[camera]["intrinsics"]
            K = _parse_intrinsic_to_K(intr)
            dist = _distortion_coeffs(intr)
            calib_h, calib_w = _image_size_hw(intr)
            if image.shape[:2] != (calib_h, calib_w):
                image = cv2.resize(image, (calib_w, calib_h), interpolation=cv2.INTER_LINEAR)
            h, w = image.shape[:2]

            mode_transforms = {
                "lidar_assumed": np.asarray(calib[camera]["extrinsics"]["cTv"], dtype=np.float64) @ vTl,
                "vehicle_assumed": np.asarray(calib[camera]["extrinsics"]["cTv"], dtype=np.float64),
            }
            canvas = image.copy()
            camera_summary = {
                "timestamp": ts,
                "camera": camera,
                "image_path": str(image_path),
                "num_gt_objects": len(frame["objects"]),
                "assumptions": {},
            }

            for mode, T in mode_transforms.items():
                mode_stats = {
                    "boxes_projected": 0,
                    "boxes_any_front": 0,
                    "boxes_inside_image": 0,
                    "suspicious_cases": 0,
                    "suspicious_reasons": defaultdict(int),
                }
                for obj in frame["objects"]:
                    corners = _corners_from_center_lwh_yaw(obj["position"], obj["dimensions"], float(obj["yaw"]))
                    uv, z_cam = _project_points(corners, T, K, dist)
                    stats = _projection_stats(uv, z_cam, (w, h))
                    obj_key = f"{ts}:{obj['gt_id']}"
                    obj_vis = object_visibility[mode][obj_key]
                    obj_vis["cameras_tested"] += 1
                    mode_stats["boxes_projected"] += 1
                    totals[mode]["boxes_projected"] += 1
                    if stats["any_front"]:
                        mode_stats["boxes_any_front"] += 1
                        totals[mode]["boxes_any_front"] += 1
                        obj_vis["any_front_cameras"] += 1
                    if stats["inside_image"]:
                        mode_stats["boxes_inside_image"] += 1
                        totals[mode]["boxes_inside_image"] += 1
                    if stats["reasonable_projected_box"]:
                        obj_vis["visible_cameras"] += 1
                    if stats["suspicious"]:
                        mode_stats["suspicious_cases"] += 1
                        totals[mode]["suspicious_cases"] += 1
                        obj_vis["suspicious_projection_cameras"] += 1
                        for reason in stats["suspicious"]:
                            mode_stats["suspicious_reasons"][reason] += 1
                            totals[mode]["suspicious_reasons"][reason] += 1
                            obj_vis["suspicious_reasons"][reason] += 1
                    if stats["inside_image"]:
                        label = f"{mode}:{obj['normalized_class']}:{obj['gt_id']}"
                        _draw_box(canvas, uv, z_cam, label, COLORS[mode])

                projected = max(1, mode_stats["boxes_projected"])
                mode_stats["percentage_inside_image"] = mode_stats["boxes_inside_image"] / projected
                mode_stats["suspicious_reasons"] = dict(mode_stats["suspicious_reasons"])
                camera_summary["assumptions"][mode] = mode_stats

            out_name = f"frame_{ts}__{camera}.jpg"
            cv2.imwrite(str(output / out_name), canvas)
            camera_summary["debug_image"] = str(output / out_name)
            frame_summaries.append(camera_summary)

    totals_plain: Dict[str, Any] = {}
    for mode, stats in totals.items():
        projected = max(1, stats["boxes_projected"])
        totals_plain[mode] = {
            "boxes_projected": int(stats["boxes_projected"]),
            "boxes_any_front": int(stats["boxes_any_front"]),
            "boxes_inside_image": int(stats["boxes_inside_image"]),
            "percentage_inside_image": float(stats["boxes_inside_image"] / projected),
            "suspicious_cases": int(stats["suspicious_cases"]),
            "suspicious_reasons": dict(stats["suspicious_reasons"]),
        }

    object_visibility_totals: Dict[str, Any] = {}
    for mode, entries_by_key in object_visibility.items():
        entries = list(entries_by_key.values())
        total_objects = len(entries)
        visible = [e for e in entries if int(e["visible_cameras"]) > 0]
        behind_all = [e for e in entries if int(e["any_front_cameras"]) == 0]
        only_suspicious = [
            e for e in entries
            if int(e["visible_cameras"]) == 0 and int(e["suspicious_projection_cameras"]) > 0
        ]
        for e in entries:
            e["suspicious_reasons"] = dict(e["suspicious_reasons"])
        top_suspicious = sorted(
            only_suspicious,
            key=lambda e: (int(e["suspicious_projection_cameras"]), -int(e["visible_cameras"])),
            reverse=True,
        )[:20]
        object_visibility_totals[mode] = {
            "total_objects_tested": int(total_objects),
            "objects_visible_any_camera": int(len(visible)),
            "any_camera_visible_pct": float(len(visible) / max(1, total_objects)),
            "average_visible_cameras_per_object": float(
                sum(int(e["visible_cameras"]) for e in entries) / max(1, total_objects)
            ),
            "objects_behind_all_cameras": int(len(behind_all)),
            "objects_only_suspicious": int(len(only_suspicious)),
            "top_suspicious_examples": top_suspicious,
        }

    summary = {
        "inputs": {
            "annotations": str(annotations),
            "calibration": str(calib_path),
            "data_root": str(data_root),
            "cameras": cameras,
            "max_frames": int(max_frames),
        },
        "warnings": [
            "This probe compares projection plausibility only; it does not prove the true coordinate frame.",
            "world_assumed is not evaluated because annotations.json does not declare a world frame transform here.",
            "Positions are treated as box centers and orientation is treated as yaw-like around +z for both tested modes.",
        ],
        "annotation_loader_warnings": ann_doc["metadata"].get("warnings", []),
        "assumption_totals": totals_plain,
        "object_visibility_totals": object_visibility_totals,
        "frames": frame_summaries,
        "missing_images": missing_images,
    }
    summary["recommendation"] = _recommend(summary)
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    visibility_summary = {
        "inputs": summary["inputs"],
        "warnings": summary["warnings"],
        "object_visibility_totals": object_visibility_totals,
        "recommendation": summary["recommendation"],
    }
    with (output / "visibility_summary.json").open("w", encoding="utf-8") as f:
        json.dump(visibility_summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project DrivIng GT boxes under multiple coordinate-frame assumptions")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/debug/annotation_projection_probe"))
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--camera-name", type=str, default="", help="Optional single camera, e.g. front_left_camera")
    return parser.parse_args()


def _print_summary(summary: Dict[str, Any]) -> None:
    print("Annotation projection probe")
    print(f"- output: {summary['inputs'].get('output', 'see --output')}")
    print(f"- cameras: {summary['inputs']['cameras']}")
    for mode, stats in summary["assumption_totals"].items():
        pct = 100.0 * float(stats["percentage_inside_image"])
        print(
            f"- {mode}: projected={stats['boxes_projected']} "
            f"front={stats['boxes_any_front']} inside={stats['boxes_inside_image']} "
            f"inside_pct={pct:.2f}% suspicious={stats['suspicious_cases']}"
        )
        if stats["suspicious_reasons"]:
            print(f"  suspicious reasons: {stats['suspicious_reasons']}")
    print("Object-level any-camera visibility:")
    for mode, stats in summary.get("object_visibility_totals", {}).items():
        pct = 100.0 * float(stats["any_camera_visible_pct"])
        print(
            f"- {mode}: objects={stats['total_objects_tested']} "
            f"visible_any_camera={stats['objects_visible_any_camera']} "
            f"visible_pct={pct:.2f}% "
            f"avg_visible_cameras={stats['average_visible_cameras_per_object']:.2f} "
            f"behind_all={stats['objects_behind_all_cameras']} "
            f"only_suspicious={stats['objects_only_suspicious']}"
        )
    if summary["missing_images"]:
        print(f"- missing/unreadable image entries: {len(summary['missing_images'])}")
    print(f"- recommendation: {summary['recommendation']}")
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")


def main() -> int:
    args = parse_args()
    summary = run_probe(
        annotations=args.annotations,
        calibration=args.calibration,
        data_root=args.data_root,
        output=args.output,
        max_frames=args.max_frames,
        camera_name=args.camera_name or None,
    )
    summary["inputs"]["output"] = str(args.output)
    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _print_summary(summary)
    print(f"Wrote summary: {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
