from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def collect_scores(per_frame_objects: List[List[Dict]]) -> List[float]:
    out: List[float] = []
    for objs in per_frame_objects:
        out.extend(float(o.get("score", 0.0)) for o in objs)
    return out


def summarize_scores(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"count": 0}
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def apply_conf_filter(per_frame_objects: List[List[Dict]], conf_thr: float) -> List[List[Dict]]:
    return [[o for o in objs if float(o.get("score", 0.0)) >= conf_thr] for objs in per_frame_objects]


def apply_per_image_top_k(per_frame_objects: List[List[Dict]], max_boxes_per_image: int) -> List[List[Dict]]:
    if max_boxes_per_image <= 0:
        return [list(objs) for objs in per_frame_objects]
    result: List[List[Dict]] = []
    for objs in per_frame_objects:
        by_cam: Dict[str, List[Dict]] = {}
        no_cam: List[Dict] = []
        for o in objs:
            cam = (o.get("bbox_2d") or {}).get("camera")
            if cam:
                by_cam.setdefault(str(cam), []).append(o)
            else:
                no_cam.append(o)
        merged: List[Dict] = []
        for cam in sorted(by_cam):
            lst = sorted(by_cam[cam], key=lambda x: float(x.get("score", 0.0)), reverse=True)
            merged.extend(lst[:max_boxes_per_image])
        merged.extend(no_cam)
        result.append(merged)
    return result


def apply_per_image_top_k_per_camera(
    per_frame_objects: List[List[Dict]],
    cam_to_max: Dict[str, int],
    default_max: int = 10_000,
) -> List[List[Dict]]:
    """每帧内按相机分组，各相机保留 score 最高的至多 cam_to_max[cam] 个（未知相机用 default_max）。"""
    result: List[List[Dict]] = []
    for objs in per_frame_objects:
        by_cam: Dict[str, List[Dict]] = {}
        no_cam: List[Dict] = []
        for o in objs:
            cam = (o.get("bbox_2d") or {}).get("camera")
            if cam:
                by_cam.setdefault(str(cam), []).append(o)
            else:
                no_cam.append(o)
        merged: List[Dict] = []
        for cam in sorted(by_cam):
            cap = int(cam_to_max.get(cam, default_max))
            lst = sorted(by_cam[cam], key=lambda x: float(x.get("score", 0.0)), reverse=True)
            if cap <= 0:
                merged.extend(lst)
            else:
                merged.extend(lst[:cap])
        merged.extend(no_cam)
        result.append(merged)
    return result


def _aabb_from_corners(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return corners.min(axis=0), corners.max(axis=0)


def _aabb_iou(c1: np.ndarray, c2: np.ndarray) -> float:
    mn1, mx1 = _aabb_from_corners(c1)
    mn2, mx2 = _aabb_from_corners(c2)
    imn = np.maximum(mn1, mn2)
    imx = np.minimum(mx1, mx2)
    d = imx - imn
    if np.any(d <= 0):
        return 0.0
    vi = float(d[0] * d[1] * d[2])
    v1 = float(np.prod(mx1 - mn1))
    v2 = float(np.prod(mx2 - mn2))
    vu = v1 + v2 - vi
    return vi / vu if vu > 0 else 0.0


def _aabb_intersection_volume(c1: np.ndarray, c2: np.ndarray) -> float:
    mn1, mx1 = _aabb_from_corners(c1)
    mn2, mx2 = _aabb_from_corners(c2)
    imn = np.maximum(mn1, mn2)
    imx = np.minimum(mx1, mx2)
    d = imx - imn
    if np.any(d <= 0):
        return 0.0
    return float(d[0] * d[1] * d[2])


def _aabb_volume(c: np.ndarray) -> float:
    mn, mx = _aabb_from_corners(c)
    d = mx - mn
    if np.any(d <= 0):
        return 0.0
    return float(d[0] * d[1] * d[2])


def _should_suppress_pair(
    kept_obj: Dict,
    kept_corners: np.ndarray,
    other_obj: Dict,
    other_corners: np.ndarray,
    iou_thr: float,
    center_near_m: float,
    size_overlap_min: float,
) -> bool:
    if _aabb_iou(kept_corners, other_corners) > iou_thr:
        return True

    c1 = np.asarray((kept_obj.get("bbox_3d_lidar") or {}).get("center"), dtype=np.float64).reshape(3)
    c2 = np.asarray((other_obj.get("bbox_3d_lidar") or {}).get("center"), dtype=np.float64).reshape(3)
    if float(np.linalg.norm(c1 - c2)) >= center_near_m:
        return False

    vi = _aabb_intersection_volume(kept_corners, other_corners)
    if vi <= 0:
        return False
    denom = min(_aabb_volume(kept_corners), _aabb_volume(other_corners))
    overlap_ratio = (vi / denom) if denom > 0 else 0.0
    return overlap_ratio >= size_overlap_min


def nms_3d(
    per_frame_objects: List[List[Dict]],
    iou_thr: float,
    center_near_m: float = 2.0,
    size_overlap_min: float = 0.3,
) -> List[List[Dict]]:
    out: List[List[Dict]] = []
    for objs in per_frame_objects:
        valid = []
        invalid = []
        for o in objs:
            corners = ((o.get("bbox_3d_lidar") or {}).get("corners"))
            if corners is None:
                invalid.append(o)
                continue
            arr = np.asarray(corners, dtype=np.float64)
            if arr.shape != (8, 3):
                invalid.append(o)
                continue
            valid.append((o, arr))
        valid_sorted = sorted(valid, key=lambda x: float(x[0].get("score", 0.0)), reverse=True)
        kept: List[Dict] = []
        kept_corners: List[np.ndarray] = []
        for obj, corners in valid_sorted:
            if all(
                not _should_suppress_pair(
                    kept_obj=kept[i],
                    kept_corners=kept_corners[i],
                    other_obj=obj,
                    other_corners=corners,
                    iou_thr=iou_thr,
                    center_near_m=center_near_m,
                    size_overlap_min=size_overlap_min,
                )
                for i in range(len(kept_corners))
            ):
                kept.append(obj)
                kept_corners.append(corners)
        out.append(kept + invalid)
    return out
