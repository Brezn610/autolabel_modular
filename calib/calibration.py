from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..settings.schemas import CalibrationBundle


def load_and_validate_calibration(calib_path: Path, cameras: List[str]) -> CalibrationBundle:
    if not calib_path.is_file():
        raise FileNotFoundError(f"找不到标定文件: {calib_path}")
    with calib_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)

    if "middle_lidar" not in calib:
        raise ValueError("calibration.json 缺少 middle_lidar")
    ml = calib["middle_lidar"]
    if "extrinsics" not in ml or "vTl" not in ml["extrinsics"]:
        raise ValueError("middle_lidar.extrinsics.vTl 缺失")

    vTl = np.asarray(ml["extrinsics"]["vTl"], dtype=np.float64)
    if vTl.shape != (4, 4):
        raise ValueError(f"vTl 必须是 4x4，当前 {vTl.shape}")

    missing: List[str] = []
    for cam in cameras:
        if cam not in calib:
            missing.append(cam)
            continue
        b = calib[cam]
        if "intrinsics" not in b or "extrinsics" not in b:
            missing.append(cam)
            continue
        if "IntrinsicMatrix" not in b["intrinsics"] or "cTv" not in b["extrinsics"]:
            missing.append(cam)
            continue
    if missing:
        raise ValueError(f"以下相机标定字段不完整: {', '.join(missing)}")

    return CalibrationBundle(raw=calib, vTl=vTl, lidar_to_vehicle=vTl)


def parse_intrinsic_to_K(intrinsics: Dict[str, Any]) -> np.ndarray:
    M = np.asarray(intrinsics["IntrinsicMatrix"], dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError(f"IntrinsicMatrix 必须是 3x3，当前 {M.shape}")
    fx = M[0, 0]
    skew = M[1, 0]
    fy = M[1, 1]
    cx = M[2, 0]
    cy = M[2, 1]
    bottom = M[2, 2]
    if abs(bottom - 1.0) > 1e-6:
        raise ValueError("IntrinsicMatrix[2,2] 预期为 1.0")
    return np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_distortion_coeffs(intrinsics: Dict[str, Any]) -> np.ndarray:
    if "DistortionCoefficients" in intrinsics:
        return np.asarray(intrinsics["DistortionCoefficients"], dtype=np.float64).reshape(-1)

    k_list = [float(x) for x in intrinsics.get("RadialDistortion", [])]
    p_list = [float(x) for x in intrinsics.get("TangentialDistortion", [])]
    if len(k_list) >= 2:
        coeffs: List[float] = [k_list[0], k_list[1], p_list[0] if p_list else 0.0, p_list[1] if len(p_list) > 1 else 0.0]
        if len(k_list) >= 3:
            coeffs.append(k_list[2])
        return np.asarray(coeffs, dtype=np.float64)
    return np.asarray([], dtype=np.float64)


def get_image_size_hw(intrinsics: Dict[str, Any]) -> Tuple[int, int]:
    h, w = intrinsics["ImageSize"]
    return int(h), int(w)


def get_T_lidar_to_cam(calib: Dict[str, Any], cam: str) -> np.ndarray:
    vTl = np.asarray(calib["middle_lidar"]["extrinsics"]["vTl"], dtype=np.float64)
    cTv = np.asarray(calib[cam]["extrinsics"]["cTv"], dtype=np.float64)
    if vTl.shape != (4, 4) or cTv.shape != (4, 4):
        raise ValueError("vTl / cTv 必须是 4x4")
    return cTv @ vTl
