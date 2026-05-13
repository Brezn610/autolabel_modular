from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def project_lidar_to_image(
    xyz_lidar: np.ndarray,
    T_l2c: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = xyz_lidar.shape[0]
    pts_h = np.hstack([xyz_lidar, np.ones((n, 1), dtype=np.float64)])
    pts_c = (T_l2c @ pts_h.T).T[:, :3].astype(np.float64)
    pts_cv = pts_c.reshape(-1, 1, 3)
    img_pts, _ = cv2.projectPoints(pts_cv, np.zeros(3), np.zeros(3), K, dist)
    # 保持 float64，避免大分辨率/远点投影时 float32 溢出导致可视化乱线
    uv = np.asarray(img_pts.reshape(-1, 2), dtype=np.float64)
    z_cam = np.asarray(pts_c[:, 2], dtype=np.float64)
    return uv, z_cam


def frustum_mask_uvz(
    uv: np.ndarray,
    z_cam: np.ndarray,
    box_xyxy: list[float],
    z_min: float,
    z_max: float,
    sam_mask_hw: Optional[np.ndarray] = None,
    *,
    sam_mask_dilate_iters: int = 1,
    sam_mask_min_points: int = 5,
    out_meta: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    视锥：2D 框 + 深度带内的 LiDAR 投影点。

    若提供 ``sam_mask_hw``（bool，形状 [H,W]，与投影 uv 同一像素坐标系），
    则再要求 ``mask[v,u]`` 为真；若交集点数 < ``sam_mask_min_points``，回退为仅用 box。
    ``sam_mask_dilate_iters``：对 mask 做 3×3 膨胀迭代次数，减轻边界漏点。
    """
    x1, y1, x2, y2 = box_xyxy
    u, v = uv[:, 0], uv[:, 1]
    base = (
        (z_cam > z_min)
        & (z_cam < z_max)
        & (u >= x1)
        & (u <= x2)
        & (v >= y1)
        & (v <= y2)
    )
    n_base = int(np.count_nonzero(base))
    if out_meta is not None:
        out_meta.clear()
        out_meta["n_box_only"] = n_base
        out_meta["used_sam_mask"] = False
        out_meta["fallback_to_box"] = False
        out_meta["n_intersection"] = 0
        out_meta["n_final"] = n_base

    if sam_mask_hw is None or not isinstance(sam_mask_hw, np.ndarray) or sam_mask_hw.size == 0:
        return base

    m = np.asarray(sam_mask_hw, dtype=np.uint8)
    if m.ndim != 2:
        return base
    m = (m > 0).astype(np.uint8)
    if int(sam_mask_dilate_iters) > 0:
        k3 = np.ones((3, 3), np.uint8)
        m = cv2.dilate(m, k3, iterations=int(sam_mask_dilate_iters))
    mb = m.astype(bool)
    H, W = mb.shape
    ui = np.clip(np.rint(u).astype(np.int32), 0, W - 1)
    vi = np.clip(np.rint(v).astype(np.int32), 0, H - 1)
    in_mask = mb[vi, ui]
    combined = base & in_mask
    n_ix = int(np.count_nonzero(combined))
    if out_meta is not None:
        out_meta["used_sam_mask"] = True
        out_meta["n_intersection"] = n_ix
    if n_ix >= int(sam_mask_min_points):
        if out_meta is not None:
            out_meta["n_final"] = n_ix
            out_meta["fallback_to_box"] = False
        return combined
    if out_meta is not None:
        out_meta["fallback_to_box"] = True
        out_meta["n_final"] = n_base
    return base
