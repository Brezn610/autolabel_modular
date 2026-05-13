"""Ego-pose 子模块：读取 DrivIng `vehicle_state/*.json` 并构造 T_world_ego / T_world_lidar。

目的：把每帧 OBB 从 LiDAR 传感器系变换到序列局部世界系，让 Temporal Refinement
（跨帧关联 + 轨迹平滑）在更稳定的参考系下工作（静止目标在世界系里近似不动）。
"""

from .ego_pose import (
    EgoPoseCache,
    build_T_world_ego,
    load_ego_pose,
    transform_boxes_to_world,
    transform_boxes_from_world,
)

__all__ = [
    "EgoPoseCache",
    "build_T_world_ego",
    "load_ego_pose",
    "transform_boxes_to_world",
    "transform_boxes_from_world",
]
