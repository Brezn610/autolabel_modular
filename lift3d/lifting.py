from __future__ import annotations

from typing import Optional

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


def fit_yaw_only_obb_from_points(pts: np.ndarray) -> o3d.geometry.OrientedBoundingBox:
    """
    在 LiDAR 坐标系下假设 z 为竖直方向，仅在水平面 (x,y) 上估计 yaw，
    再对旋转后的点做轴对齐包围盒，得到与路面一致的竖直长方体（无任意 roll/pitch）。
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd.get_oriented_bounding_box()

    c0 = pts.mean(axis=0)
    xy = pts[:, :2] - c0[:2]
    if float(np.max(np.std(xy, axis=0))) < 1e-6:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd.get_oriented_bounding_box()

    c_xy = np.cov(xy.T)
    if not np.all(np.isfinite(c_xy)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd.get_oriented_bounding_box()

    evals, evecs = np.linalg.eigh(c_xy)
    main = evecs[:, int(np.argmax(evals))]
    if float(np.linalg.norm(main)) < 1e-9:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd.get_oriented_bounding_box()

    yaw = float(np.arctan2(main[1], main[0]))
    cy, sy = np.cos(yaw), np.sin(yaw)
    r = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rel = pts - c0
    local = (r.T @ rel.T).T
    lo = local.min(axis=0)
    hi = local.max(axis=0)
    extent = np.maximum(hi - lo, 1e-3)
    center_local = (lo + hi) * 0.5
    center = c0 + r @ center_local
    return o3d.geometry.OrientedBoundingBox(center, r, extent)


def cluster_and_fit_obb(
    xyz_lidar: np.ndarray,
    eps: float,
    min_samples: int,
    yaw_only: bool = True,
) -> Optional[o3d.geometry.OrientedBoundingBox]:
    if xyz_lidar.shape[0] < min_samples:
        return None
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz_lidar)
    labels = clustering.labels_
    best_label = None
    best_count = -1
    for lb in sorted(set(labels)):
        if lb == -1:
            continue
        cnt = int(np.sum(labels == lb))
        if cnt > best_count:
            best_count = cnt
            best_label = lb
    if best_label is None:
        return None
    pts = xyz_lidar[labels == best_label]
    if yaw_only:
        return fit_yaw_only_obb_from_points(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd.get_oriented_bounding_box()


def obb_to_json_dict(obb: o3d.geometry.OrientedBoundingBox, class_name: str, score: float) -> dict:
    Rm = np.asarray(obb.R, dtype=np.float64)
    yaw = float(np.arctan2(Rm[1, 0], Rm[0, 0]))
    center = np.asarray(obb.center, dtype=np.float64)
    extent = np.asarray(obb.extent, dtype=np.float64)
    corners = np.asarray(obb.get_box_points(), dtype=np.float64)
    return {
        "category": class_name,
        "score": float(score),
        "bbox_3d_lidar": {
            "center": center.tolist(),
            "extent": extent.tolist(),
            "yaw_rad": yaw,
            "R": Rm.tolist(),
            "corners": corners.tolist(),
        },
    }
