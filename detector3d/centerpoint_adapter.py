"""Placeholder for future CenterPoint integration.

Future CenterPoint outputs should be converted into the canonical schema defined
in ``detector3d.schema``:

    [x, y, z, length, width, height, yaw] in a LiDAR-frame-compatible coordinate
    convention, plus ``score_3d``, normalized class, and source="centerpoint".

This module intentionally does not import MMDetection3D or any heavy runtime.
Downstream tracking/projection/matching/fusion should consume the canonical JSON
schema and should not care whether boxes came from the mock provider, CenterPoint,
TransFusion, or MS3D.
"""

from __future__ import annotations


class CenterPointAdapter:
    def __init__(self) -> None:
        raise NotImplementedError("CenterPoint adapter is a placeholder; convert outputs to detector3d.schema later.")
