from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# 包根目录 autolabel_modular/（settings 的上一级）
_PKG_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_ROOT = Path(os.environ.get("DEMO_DATA_ROOT", "/mnt/d/thesis/day"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("DEMO_OUT", str(_PKG_ROOT / "out")))


@dataclass
class AppConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    max_frames: int = 100
    cameras: List[str] = field(
        default_factory=lambda: [
            "front_left_camera",
            "front_right_camera",
            "back_left_camera",
            "back_right_camera",
            "left_camera",
            "right_camera",
        ]
    )
    cameras_for_detection: List[str] = field(default_factory=list)
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-base"
    dino_text_prompt: str = "car. truck. bus. pedestrian. bicycle. motorcycle."
    dino_box_threshold: float = 0.22
    dino_text_threshold: float = 0.22
    # True：按 DINO_CAMERA_CONFIG 使用每相机 prompt + 低阈值召回 + Top-K；False：用上面固定 prompt/threshold
    dino_use_camera_specific: bool = True
    # 相机专项模式下的召回阈值（尽量多框，再由每相机 top_k 截断）
    dino_recall_box_threshold: float = 0.01
    dino_recall_text_threshold: float = 0.01
    # SAM 2：诊断（可视化 + raw_dino_debug）；可与 sam_enabled_in_frustum 组合
    sam2_debug_enabled: bool = False
    sam2_model_id: str = "facebook/sam2-hiera-small"
    sam2_mask_threshold: float = 0.5
    sam2_debug_store_compressed_mask: bool = False
    # 方案 A：SAM mask 参与视锥取点（< min 点则回退为仅用 2D box）
    sam_enabled_in_frustum: bool = False
    sam_frustum_fallback_min_points: int = 5
    sam_frustum_mask_dilate_iters: int = 1
    sam_debug_max_points_draw: int = 8000
    # Temporal Refinement（参考 MS3D darrenjkt/MS3D 中 temporal_refinement.py + tracker/）
    # 在 3D NMS 之后对最近 N 帧的 OBB 做跨帧/跨相机关联 + 轨迹平滑 + 轨迹内去重。
    temporal_enabled: bool = False
    temporal_window_frames: int = 7                 # track 允许连续丢失帧数
    temporal_iou_threshold: float = 0.5             # BEV IoU 关联门限
    temporal_smoothing_method: str = "moving_average"  # "moving_average" 或 "spline"
    temporal_min_dets_per_track: int = 3            # 低于该值的 track 不平滑、打标后保留
    temporal_center_gate_m: float = 4.0             # BEV 中心距剪枝阈值
    temporal_debug_vis: bool = True                 # 是否输出 temporal_debug/ BEV 可视化
    # Ego-pose 补偿：从 vehicle_state/*.json 取每帧 ego 位姿，先把 OBB 变到“序列局部世界系”
    # 再做 Temporal Refinement（降低自车运动对跨帧关联的破坏）。
    ego_pose_enabled: bool = False
    ego_pose_dir_name: str = "vehicle_state"
    ego_pose_max_dt_ns: int = 60_000_000            # 时间戳最大可接受偏差（ns），10Hz 周期 100ms
    ego_pose_use_first_frame_origin: bool = True     # 把首帧 ego 位置作为 world 原点
    ego_pose_angle_in_degrees: bool = True           # DrivIng 的 roll/pitch/yaw 以 **度** 存储
    max_points_projection: int = 50_000
    frustum_z_min: float = 0.5
    frustum_z_max: float = 120.0
    dbscan_eps: float = 0.6
    dbscan_min_samples: int = 8
    # True：在 x–y 平面估 yaw、z 与 LiDAR z 对齐（车体/路面更自然）；False：Open3D 最小体积 OBB（易歪）
    obb_yaw_only: bool = True
    class_colors_bgr: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            "car": (60, 20, 220),
            "truck": (200, 120, 0),
            "bus": (0, 200, 200),
            "pedestrian": (200, 0, 200),
            "bicycle": (0, 255, 128),
            "motorcycle": (128, 128, 255),
        }
    )

    def __post_init__(self) -> None:
        if not self.cameras_for_detection:
            self.cameras_for_detection = list(self.cameras)
