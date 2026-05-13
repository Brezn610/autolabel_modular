"""配置与数据结构。"""
from __future__ import annotations

from .config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, AppConfig
from .schemas import CalibrationBundle, FrameRecord

__all__ = [
    "AppConfig",
    "CalibrationBundle",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "FrameRecord",
]
