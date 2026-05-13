"""2D 检测（Grounding DINO）。"""
from __future__ import annotations

from .detection import grounding_dino_detect, load_grounding_dino, map_label_to_class, normalize_label_name

__all__ = ["grounding_dino_detect", "load_grounding_dino", "map_label_to_class", "normalize_label_name"]
