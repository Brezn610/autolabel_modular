"""LiDAR–多相机 2D 检测 → 视锥 lifting → 3D OBB → 后处理 → JSON / 报告。"""

from __future__ import annotations

from autolabel_modular.pipeline.runner import run_pipeline
from autolabel_modular.settings.config import AppConfig, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT

__all__ = [
    "AppConfig",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "run_pipeline",
]
