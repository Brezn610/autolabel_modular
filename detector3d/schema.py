from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


VALID_CLASSES = {"vehicle", "pedestrian"}
VALID_SOURCES = {"mock_lidar_detector", "centerpoint", "transfusion", "ms3d"}

COORDINATE_FRAME_NOTE = (
    "annotations treated as LiDAR-like based on BEV support diagnostics; "
    "not explicitly declared by dataset"
)


@dataclass
class Detection3D:
    det_id: str
    class_name: str
    box3d_lidar: List[float]
    score_3d: float
    source: str
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_debug_fields: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "det_id": self.det_id,
            "class": self.class_name,
            "box3d_lidar": [float(x) for x in self.box3d_lidar],
            "score_3d": float(self.score_3d),
            "source": self.source,
        }
        if include_debug_fields and self.debug:
            out.update(self.debug)
        return out


@dataclass
class FrameDetections3D:
    frame_id: str
    timestamp: int
    boxes3d: List[Detection3D] = field(default_factory=list)

    def to_dict(self, *, include_debug_fields: bool = True) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": int(self.timestamp),
            "boxes3d": [box.to_dict(include_debug_fields=include_debug_fields) for box in self.boxes3d],
        }


@dataclass
class Detections3DDocument:
    source: str
    frames: List[FrameDetections3D]
    coordinate_frame: str = "lidar_like"
    coordinate_frame_note: str = COORDINATE_FRAME_NOTE
    warnings: List[str] = field(default_factory=list)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_debug_fields: bool = True) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "source": self.source,
            "coordinate_frame": self.coordinate_frame,
            "coordinate_frame_note": self.coordinate_frame_note,
            "warnings": list(self.warnings),
        }
        metadata.update(self.extra_metadata)
        return {
            "metadata": metadata,
            "frames": [frame.to_dict(include_debug_fields=include_debug_fields) for frame in self.frames],
        }

