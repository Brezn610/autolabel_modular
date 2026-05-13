"""端到端编排：检测 → 视锥 → 聚类 → 后处理 → 落盘。"""
from __future__ import annotations

from .runner import run_pipeline

__all__ = ["run_pipeline"]
