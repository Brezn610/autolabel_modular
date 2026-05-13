"""Grounding DINO：按相机的 prompt 与 Top-K（检测阶段低阈值召回后按分数截断）。"""
from __future__ import annotations

from typing import Any, Dict, Tuple

# 键须与 AppConfig.cameras / timesync 中的相机名一致（含 _camera 后缀）
DINO_CAMERA_CONFIG: Dict[str, Dict[str, Any]] = {
    "front_left_camera": {
        "top_k": 6,
        "prompt": "car . truck . bus . vehicle . suv",
    },
    "front_right_camera": {
        "top_k": 5,
        "prompt": "car . truck . bus . vehicle . suv",
    },
    "left_camera": {
        "top_k": 6,
        "prompt": "car . truck . bus . vehicle . suv side view",
    },
    "right_camera": {
        "top_k": 5,
        "prompt": "car . truck . bus . vehicle . suv side view",
    },
    "back_left_camera": {
        "top_k": 8,
        "prompt": "car . truck . bus . vehicle . suv . rear view . behind . distant vehicle",
    },
    "back_right_camera": {
        "top_k": 8,
        "prompt": "car . truck . bus . vehicle . suv . rear view . behind . distant vehicle",
    },
}

_FALLBACK_CAM = "front_left_camera"


def dino_prompt_and_top_k(cam: str) -> Tuple[str, int]:
    row = DINO_CAMERA_CONFIG.get(cam) or DINO_CAMERA_CONFIG[_FALLBACK_CAM]
    return str(row["prompt"]), int(row["top_k"])


def camera_top_k_map() -> Dict[str, int]:
    return {k: int(v["top_k"]) for k, v in DINO_CAMERA_CONFIG.items()}
