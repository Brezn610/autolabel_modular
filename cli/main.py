#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ..pipeline.runner import run_pipeline
from ..reports.console import print_report
from ..settings.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, AppConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Modular auto-label pipeline")
    ap.add_argument("--data-root", type=str, default="", help="数据根目录，默认读取 DEMO_DATA_ROOT 或内置默认值")
    ap.add_argument("--output-root", type=str, default="", help="输出目录，默认读取 DEMO_OUT 或内置默认值")
    ap.add_argument("--conf-thr", type=float, default=0.18)
    ap.add_argument("--max-boxes-per-image", type=int, default=8)
    ap.add_argument("--nms-iou-thr", type=float, default=0.3)
    ap.add_argument("--nms-center-dist-m", type=float, default=2.0)
    ap.add_argument("--nms-size-overlap-min", type=float, default=0.3)
    ap.add_argument(
        "--save-projection-vis",
        action="store_true",
        help="保存 LiDAR 到相机的投影可视化（会增加运行时间）",
    )
    ap.add_argument(
        "--single-camera",
        type=str,
        default="",
        help="仅在一个相机上做检测诊断，例如 front_left_camera",
    )
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument(
        "--no-obb-yaw-only",
        action="store_true",
        help="关闭仅 yaw 竖直盒，改用 Open3D 最小体积 OBB（旧行为，易出现歪框）",
    )
    ap.add_argument(
        "--no-dino-camera-specific",
        action="store_true",
        help="关闭按相机 prompt + Top-K，改回全局 dino_text_prompt 与 box/text 阈值",
    )
    ap.add_argument(
        "--sam2-debug",
        action="store_true",
        help="DINO 后接 SAM2：raw_dino_debug SAM 字段 + sam_debug 图（可与 --sam-frustum 联用）",
    )
    ap.add_argument(
        "--sam-frustum",
        action="store_true",
        help="SAM mask 参与 LiDAR 视锥取点（回退阈值见 AppConfig.sam_frustum_fallback_min_points）",
    )
    ap.add_argument(
        "--temporal",
        action="store_true",
        help="启用 Temporal Refinement（参考 MS3D）：3D NMS 后做跨帧跨相机关联+平滑+去重",
    )
    ap.add_argument(
        "--temporal-window",
        type=int,
        default=-1,
        help="Temporal track 允许连续丢失的帧数（默认读 AppConfig.temporal_window_frames）",
    )
    ap.add_argument(
        "--temporal-iou",
        type=float,
        default=-1.0,
        help="Temporal 关联的 BEV IoU 门限（默认读 AppConfig.temporal_iou_threshold）",
    )
    ap.add_argument(
        "--temporal-min-dets",
        type=int,
        default=-1,
        help="track 最小长度；低于该值的 track 不做平滑（默认读 AppConfig.temporal_min_dets_per_track）",
    )
    ap.add_argument(
        "--temporal-smoothing",
        type=str,
        default="",
        choices=["", "moving_average", "spline"],
        help="平滑方法：moving_average 或 spline",
    )
    ap.add_argument(
        "--ego-pose",
        action="store_true",
        help="使用 vehicle_state/ 里每帧 ego 位姿把 OBB 先变到序列局部世界系再做 Temporal Refinement",
    )
    ap.add_argument(
        "--ego-pose-max-dt-ms",
        type=float,
        default=-1.0,
        help="ego 位姿最大时间偏差（ms，默认读 AppConfig.ego_pose_max_dt_ns）",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = AppConfig(
        data_root=Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT,
        output_root=Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT,
        max_frames=args.max_frames,
    )
    if args.single_camera:
        cfg.cameras_for_detection = [args.single_camera]
    cfg.obb_yaw_only = not args.no_obb_yaw_only
    cfg.dino_use_camera_specific = not args.no_dino_camera_specific
    cfg.sam2_debug_enabled = bool(args.sam2_debug)
    cfg.sam_enabled_in_frustum = bool(args.sam_frustum)
    cfg.temporal_enabled = bool(args.temporal)
    if args.temporal_window and args.temporal_window > 0:
        cfg.temporal_window_frames = int(args.temporal_window)
    if args.temporal_iou and args.temporal_iou > 0:
        cfg.temporal_iou_threshold = float(args.temporal_iou)
    if args.temporal_min_dets and args.temporal_min_dets > 0:
        cfg.temporal_min_dets_per_track = int(args.temporal_min_dets)
    if args.temporal_smoothing:
        cfg.temporal_smoothing_method = str(args.temporal_smoothing)
    cfg.ego_pose_enabled = bool(args.ego_pose)
    if args.ego_pose_max_dt_ms and args.ego_pose_max_dt_ms > 0:
        cfg.ego_pose_max_dt_ns = int(float(args.ego_pose_max_dt_ms) * 1_000_000)

    stats = run_pipeline(
        cfg=cfg,
        conf_thr=args.conf_thr,
        max_boxes_per_image=args.max_boxes_per_image,
        nms_iou_thr=args.nms_iou_thr,
        nms_center_dist_m=args.nms_center_dist_m,
        nms_size_overlap_min=args.nms_size_overlap_min,
        save_projection_vis=args.save_projection_vis,
    )
    paths = stats.get("paths", {})
    raw_debug_path = paths.get("raw_dino_debug")
    stage_stats_path = str(cfg.output_root / "stage_stats.json")
    if raw_debug_path:
        print_report(
            raw_dino_debug_path=Path(raw_debug_path),
            stage_stats_path=Path(stage_stats_path),
        )
    print("\n完成，关键统计：")
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
