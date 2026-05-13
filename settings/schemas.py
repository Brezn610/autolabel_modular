from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_ns: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)


@dataclass
class CalibrationBundle:
    raw: Dict[str, Any]
    vTl: np.ndarray
    lidar_to_vehicle: np.ndarray
