"""SAM mask 与 LiDAR 视锥结合：resize / RLE 解码、与 frustum_mask_uvz 衔接、调试图。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..geom.projection import frustum_mask_uvz
from .sam2_masks import cv2_resize_bool, draw_sam_overlay_bgr


def decode_sam_mask_optional(mask_or_rle: Union[np.ndarray, Dict[str, Any], None]) -> Optional[np.ndarray]:
    """
    接受 bool/uint8 [H,W] 或 COCO 式 RLE dict（需 pycocotools）。
    无法解析时返回 None。
    """
    if mask_or_rle is None:
        return None
    if isinstance(mask_or_rle, np.ndarray):
        if mask_or_rle.size == 0:
            return None
        return mask_or_rle.astype(bool)
    if isinstance(mask_or_rle, dict) and "counts" in mask_or_rle:
        try:
            from pycocotools import mask as mask_utils  # type: ignore[import-untyped]

            return mask_utils.decode(mask_or_rle).astype(bool)
        except Exception:
            return None
    return None


def apply_sam_mask_to_frustum(
    uv: np.ndarray,
    z_cam: np.ndarray,
    box_xyxy: List[float],
    z_min: float,
    z_max: float,
    sam_mask_hw: Union[np.ndarray, Dict[str, Any], None],
    calib_hw: Tuple[int, int],
    *,
    dilate_iters: int = 1,
    min_points: int = 5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    将 SAM bool mask 对齐到标定图像高宽后，调用带 mask 分支的 ``frustum_mask_uvz``。

    返回 (点布尔掩码, meta)，meta 含是否使用 mask、是否回退、box-only / 最终点数等。
    """
    h_calib, w_calib = int(calib_hw[0]), int(calib_hw[1])
    meta: Dict[str, Any] = {
        "used_sam_mask": False,
        "fallback_to_box": False,
        "n_box_only": 0,
        "n_intersection": 0,
        "n_final": 0,
    }
    m = decode_sam_mask_optional(sam_mask_hw)  # ndarray 或 COCO RLE（需 pycocotools）
    if m is not None and isinstance(m, np.ndarray) and m.size > 0:
        if m.shape[0] != h_calib or m.shape[1] != w_calib:
            m = cv2_resize_bool(m, (w_calib, h_calib))
    else:
        m = None

    sub: Dict[str, Any] = {}
    mask = frustum_mask_uvz(
        uv,
        z_cam,
        box_xyxy,
        z_min,
        z_max,
        m,
        sam_mask_dilate_iters=int(dilate_iters),
        sam_mask_min_points=int(min_points),
        out_meta=sub,
    )
    meta.update(sub)
    meta["used_sam_mask"] = bool(sub.get("used_sam_mask", False))
    meta["fallback_to_box"] = bool(sub.get("fallback_to_box", False))
    return mask, meta


def draw_sam_frustum_debug_bgr(
    image_bgr: np.ndarray,
    boxes_xyxy: List[List[float]],
    sam_entries: List[Dict[str, Any]],
    uv_frustum_per_det: List[np.ndarray],
    point_color_bgr: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """DINO 框 + SAM mask 着色 + 最终视锥内 LiDAR 投影点（每检测一条列表，可为空数组）。"""
    canvas = draw_sam_overlay_bgr(image_bgr, boxes_xyxy, sam_entries)
    h, w = canvas.shape[:2]
    for uv in uv_frustum_per_det:
        if uv is None or uv.size == 0:
            continue
        for i in range(uv.shape[0]):
            u, v = int(round(float(uv[i, 0]))), int(round(float(uv[i, 1])))
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(canvas, (u, v), 1, point_color_bgr, -1, lineType=cv2.LINE_AA)
    return canvas
