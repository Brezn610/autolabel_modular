from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from dataset.annotation_projection_probe import (
    BOX_EDGES,
    DEFAULT_CAMERAS,
    _camera_image_path,
    _corners_from_center_lwh_yaw,
    _distortion_coeffs,
    _draw_box,
    _image_size_hw,
    _load_calibration,
    _load_timesync,
    _parse_intrinsic_to_K,
    _project_points,
    _resolve_calibration_path,
)


COLOR_BY_CLASS = {
    "vehicle": (0, 180, 255),
    "pedestrian": (80, 220, 80),
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def _available_cameras(calib: Dict[str, Any], data_root: Path) -> List[str]:
    cameras: List[str] = []
    for camera in DEFAULT_CAMERAS:
        if camera in calib and (data_root / camera).is_dir():
            cameras.append(camera)
    if cameras:
        return cameras
    return [
        key for key, value in calib.items()
        if isinstance(value, dict)
        and "intrinsics" in value
        and "extrinsics" in value
        and "cTv" in value.get("extrinsics", {})
    ]


def _project_box_to_camera(
    box3d_lidar: List[float],
    T_lidar_to_camera: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    image_wh: Tuple[int, int],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    x, y, z, length, width, height, yaw = [float(v) for v in box3d_lidar]
    corners = _corners_from_center_lwh_yaw([x, y, z], [length, width, height], yaw)
    uv, z_cam = _project_points(corners, T_lidar_to_camera, K, dist)
    w, h = image_wh
    in_front = z_cam > 0.05
    num_front = int(np.count_nonzero(in_front))
    if num_front < 2:
        return {
            "in_fov": False,
            "projected_box2d": None,
            "num_corners_in_front": num_front,
            "num_corners_inside_image": 0,
            "unclipped_box2d": None,
            "suspicious": ["most_corners_behind_camera"],
        }, uv, z_cam

    pts = uv[in_front]
    finite = np.isfinite(pts).all(axis=1)
    if not np.any(finite):
        return {
            "in_fov": False,
            "projected_box2d": None,
            "num_corners_in_front": num_front,
            "num_corners_inside_image": 0,
            "unclipped_box2d": None,
            "suspicious": ["non_finite_projection"],
        }, uv, z_cam

    pts = pts[finite]
    x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
        & in_front
        & np.isfinite(uv).all(axis=1)
    )
    num_inside = int(np.count_nonzero(inside))
    intersects = x2 >= 0 and y2 >= 0 and x1 < w and y1 < h
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area_ratio = (bw * bh) / max(1.0, float(w * h))
    suspicious: List[str] = []
    if num_front < 4:
        suspicious.append("few_corners_in_front")
    if area_ratio > 1.5 or bw > 3.0 * w or bh > 3.0 * h:
        suspicious.append("extremely_large_projected_box")
    if not intersects:
        suspicious.append("outside_image")

    if not intersects:
        return {
            "in_fov": False,
            "projected_box2d": None,
            "num_corners_in_front": num_front,
            "num_corners_inside_image": num_inside,
            "unclipped_box2d": [x1, y1, x2, y2],
            "suspicious": suspicious,
        }, uv, z_cam

    clipped = [
        float(np.clip(x1, 0, max(0, w - 1))),
        float(np.clip(y1, 0, max(0, h - 1))),
        float(np.clip(x2, 0, max(0, w - 1))),
        float(np.clip(y2, 0, max(0, h - 1))),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        suspicious.append("non_positive_clipped_box")
        return {
            "in_fov": False,
            "projected_box2d": None,
            "num_corners_in_front": num_front,
            "num_corners_inside_image": num_inside,
            "unclipped_box2d": [x1, y1, x2, y2],
            "suspicious": suspicious,
        }, uv, z_cam

    return {
        "in_fov": True,
        "projected_box2d": clipped,
        "num_corners_in_front": num_front,
        "num_corners_inside_image": num_inside,
        "unclipped_box2d": [x1, y1, x2, y2],
        "suspicious": suspicious,
    }, uv, z_cam


def project_tracks(
    *,
    tracks_path: Path,
    calibration_path: Path,
    data_root: Path,
    output_path: Path,
    debug_images: bool = False,
    debug_dir: Path = Path("outputs/debug/track_projection_probe"),
    max_debug_frames: int = 10,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tracks_doc = _read_json(tracks_path)
    calib_path = _resolve_calibration_path(calibration_path, data_root)
    calib = _load_calibration(calib_path)
    timesync = _load_timesync(data_root)
    cameras = _available_cameras(calib, data_root)
    if not cameras:
        raise ValueError("No cameras found in calibration/data root")

    vTl = np.asarray(calib["middle_lidar"]["extrinsics"]["vTl"], dtype=np.float64)
    camera_params: Dict[str, Dict[str, Any]] = {}
    for camera in cameras:
        intr = calib[camera]["intrinsics"]
        camera_params[camera] = {
            "K": _parse_intrinsic_to_K(intr),
            "dist": _distortion_coeffs(intr),
            "image_hw": _image_size_hw(intr),
            "T_lidar_to_camera": np.asarray(calib[camera]["extrinsics"]["cTv"], dtype=np.float64) @ vTl,
        }

    frames_out: List[Dict[str, Any]] = []
    per_camera_counts: Dict[str, Counter[str]] = {camera: Counter() for camera in cameras}
    total_objects = 0
    total_projection_pairs = 0
    total_in_fov = 0
    suspicious_count = 0

    debug_dir_out = ""
    if debug_images:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_dir_out = str(debug_dir)

    for frame_index, frame in enumerate(tracks_doc.get("frames") or []):
        timestamp = int(frame["timestamp"])
        frame_objects: List[Dict[str, Any]] = []
        for obj in frame.get("objects") or []:
            total_objects += 1
            projections: Dict[str, Any] = {}
            for camera in cameras:
                params = camera_params[camera]
                calib_h, calib_w = params["image_hw"]
                proj, _uv, _zc = _project_box_to_camera(
                    [float(x) for x in obj["box3d_lidar"]],
                    params["T_lidar_to_camera"],
                    params["K"],
                    params["dist"],
                    (calib_w, calib_h),
                )
                projections[camera] = {
                    "in_fov": bool(proj["in_fov"]),
                    "projected_box2d": proj["projected_box2d"],
                    "num_corners_in_front": int(proj["num_corners_in_front"]),
                    "num_corners_inside_image": int(proj["num_corners_inside_image"]),
                }
                if proj.get("suspicious"):
                    projections[camera]["suspicious"] = proj["suspicious"]
                    suspicious_count += 1
                total_projection_pairs += 1
                per_camera_counts[camera]["total"] += 1
                if proj["in_fov"]:
                    total_in_fov += 1
                    per_camera_counts[camera]["in_fov"] += 1
                else:
                    per_camera_counts[camera]["not_in_fov"] += 1

            frame_objects.append(
                {
                    "track_id": int(obj["track_id"]),
                    "det_id": str(obj["det_id"]),
                    "class": str(obj["class"]),
                    "box3d_lidar": [float(x) for x in obj["box3d_lidar"]],
                    "score_3d": float(obj.get("score_3d", 0.0)),
                    "camera_projections": projections,
                }
            )

        frames_out.append(
            {
                "frame_id": str(frame["frame_id"]),
                "timestamp": timestamp,
                "objects": frame_objects,
            }
        )

        if debug_images and frame_index < int(max_debug_frames):
            _write_debug_images(
                frame=frames_out[-1],
                raw_frame=frame,
                cameras=cameras,
                camera_params=camera_params,
                timesync=timesync,
                data_root=data_root,
                debug_dir=debug_dir,
            )

    summary = {
        "frames_processed": len(frames_out),
        "track_objects_processed": total_objects,
        "num_cameras": len(cameras),
        "total_object_camera_projections": total_projection_pairs,
        "projections_in_fov": total_in_fov,
        "in_fov_percentage": float(total_in_fov / max(1, total_projection_pairs)),
        "per_camera_in_fov_counts": {
            camera: {
                "total": int(counts["total"]),
                "in_fov": int(counts["in_fov"]),
                "not_in_fov": int(counts["not_in_fov"]),
                "in_fov_percentage": float(counts["in_fov"] / max(1, counts["total"])),
            }
            for camera, counts in per_camera_counts.items()
        },
        "suspicious_projections": int(suspicious_count),
        "debug_image_dir": debug_dir_out,
    }
    out_doc = {
        "metadata": {
            "source": "project_tracks",
            "input_tracks": str(tracks_path),
            "calibration": str(calib_path),
            "data_root": str(data_root),
            "coordinate_frame": "lidar_like",
            "note": "3D boxes are projected using existing calibration; coordinate frame was inferred from BEV diagnostics",
            "cameras": cameras,
            "summary": summary,
        },
        "frames": frames_out,
    }
    validate_projection_doc(out_doc, cameras)
    _write_json(output_path, out_doc)
    return out_doc, summary


def _write_debug_images(
    *,
    frame: Dict[str, Any],
    raw_frame: Dict[str, Any],
    cameras: List[str],
    camera_params: Dict[str, Dict[str, Any]],
    timesync: Dict[int, Dict[str, str]],
    data_root: Path,
    debug_dir: Path,
) -> None:
    timestamp = int(frame["timestamp"])
    files = timesync.get(timestamp) or {}
    for camera in cameras:
        image_name = files.get(camera)
        if not image_name:
            continue
        image_path = _camera_image_path(data_root, camera, image_name)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h_calib, w_calib = camera_params[camera]["image_hw"]
        if image.shape[:2] != (h_calib, w_calib):
            image = cv2.resize(image, (w_calib, h_calib), interpolation=cv2.INTER_LINEAR)
        canvas = image.copy()
        for obj in frame["objects"]:
            proj = obj["camera_projections"][camera]
            if not proj["in_fov"]:
                continue
            uv, zc = _project_uv_for_debug(obj["box3d_lidar"], camera_params[camera])
            label = f"t{obj['track_id']} {obj['class']} {obj.get('score_3d', 0.0):.2f}"
            color = COLOR_BY_CLASS.get(str(obj["class"]), (255, 255, 255))
            _draw_box(canvas, uv, zc, label, color)
        out_name = f"frame_{frame['frame_id']}__{camera}.jpg"
        cv2.imwrite(str(debug_dir / out_name), canvas)


def _project_uv_for_debug(box3d_lidar: List[float], params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z, length, width, height, yaw = [float(v) for v in box3d_lidar]
    corners = _corners_from_center_lwh_yaw([x, y, z], [length, width, height], yaw)
    return _project_points(corners, params["T_lidar_to_camera"], params["K"], params["dist"])


def validate_projection_doc(doc: Dict[str, Any], cameras: List[str]) -> List[str]:
    errors: List[str] = []
    if "metadata" not in doc or not isinstance(doc["metadata"], dict):
        errors.append("missing metadata")
    frames = doc.get("frames")
    if not isinstance(frames, list):
        errors.append("frames must be a list")
        return errors
    for frame_idx, frame in enumerate(frames):
        for field in ("frame_id", "timestamp", "objects"):
            if field not in frame:
                errors.append(f"frames[{frame_idx}] missing {field}")
        for obj_idx, obj in enumerate(frame.get("objects") or []):
            prefix = f"frames[{frame_idx}].objects[{obj_idx}]"
            for field in ("track_id", "det_id", "class", "box3d_lidar", "camera_projections"):
                if field not in obj:
                    errors.append(f"{prefix} missing {field}")
            projections = obj.get("camera_projections")
            if not isinstance(projections, dict):
                errors.append(f"{prefix}.camera_projections must be object")
                continue
            for camera in cameras:
                proj = projections.get(camera)
                if not isinstance(proj, dict):
                    errors.append(f"{prefix}.camera_projections missing {camera}")
                    continue
                if "in_fov" not in proj or "projected_box2d" not in proj:
                    errors.append(f"{prefix}.{camera} missing projection fields")
                    continue
                if proj["in_fov"]:
                    box = proj["projected_box2d"]
                    if not isinstance(box, list) or len(box) != 4:
                        errors.append(f"{prefix}.{camera}.projected_box2d must have 4 numbers when in_fov=true")
                elif proj["projected_box2d"] is not None:
                    errors.append(f"{prefix}.{camera}.projected_box2d must be null when in_fov=false")
    if errors:
        raise ValueError("invalid projection output:\n" + "\n".join(errors[:20]))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 3D track boxes into camera images")
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-images", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/debug/track_projection_probe"))
    parser.add_argument("--max-debug-frames", type=int, default=10)
    return parser.parse_args()


def _print_summary(summary: Dict[str, Any], output: Path) -> None:
    print("Track projection summary")
    print(f"- frames processed: {summary['frames_processed']}")
    print(f"- track objects processed: {summary['track_objects_processed']}")
    print(f"- cameras: {summary['num_cameras']}")
    print(f"- total object-camera projections: {summary['total_object_camera_projections']}")
    print(f"- projections in_fov: {summary['projections_in_fov']}")
    print(f"- in_fov percentage: {100.0 * summary['in_fov_percentage']:.2f}%")
    print(f"- suspicious projections: {summary['suspicious_projections']}")
    print("- per-camera in_fov counts:")
    for camera, counts in summary["per_camera_in_fov_counts"].items():
        print(
            f"  {camera}: {counts['in_fov']}/{counts['total']} "
            f"({100.0 * counts['in_fov_percentage']:.2f}%)"
        )
    print(f"- output: {output}")
    if summary.get("debug_image_dir"):
        print(f"- debug images: {summary['debug_image_dir']}")


def main() -> int:
    args = parse_args()
    doc, summary = project_tracks(
        tracks_path=args.tracks,
        calibration_path=args.calibration,
        data_root=args.data_root,
        output_path=args.output,
        debug_images=args.debug_images,
        debug_dir=args.debug_dir,
        max_debug_frames=args.max_debug_frames,
    )
    _print_summary(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
