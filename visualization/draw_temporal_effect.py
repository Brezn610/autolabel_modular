#!/usr/bin/env python3
"""时序精炼效果可视化（Temporal Refinement effect viz）。

输入
----
- ``out/annotations_demo_raw.json``  : raw_kept，未经 conf/Top-K/NMS/temporal；做为 *pre* 状态来源
- ``out/annotations_demo.json``      : final_objects，已做 NMS (±temporal ±ego)；做为 *post* 状态
- ``out/stage_stats.json``           : 取 temporal_stats 做汇总
- ``out/raw_dino_debug.json``        : 取 temporal_meta / 回写的 temporal_track_id
- ``<data_root>/vehicle_state/``     : ego 位姿（可选；用于把轨迹画在序列局部 world 系下）

为了保证 "pre temporal" 是真正的对照，我们在脚本里 **再跑一遍** 与 runner 一致的
置信度过滤 + per-camera Top-K + 3D NMS（不含 temporal/ego），从 raw_kept 重建出
``after_nms`` 状态。这样：
    pre  = 重新跑 conf+topk+nms 后的结果（无 temporal）
    post = annotations_demo.json（有 temporal±ego）

输出
----
- ``out/temporal_effect/bev_frame_XXXXXX.jpg`` 每帧左右对比：左=pre 右=post，按 track 上色
- ``out/temporal_effect/trajectories_bev.jpg`` 全序列 track 轨迹（world 系；若无 ego 用 LiDAR 系）
- ``out/temporal_effect/tracks_overview.html`` 汇总 HTML：统计 + 缩略图 + 轨迹图

运行示例
--------
cd /home/chase_610
python -m autolabel_modular.visualization.draw_temporal_effect \
    --output-subdir temporal_effect \
    --max-frames 0
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from autolabel_modular.calib.calibration import load_and_validate_calibration
from autolabel_modular.ego_pose.ego_pose import (
    EgoPoseCache,
    build_per_frame_T_world_lidar,
    transform_boxes_to_world,
)
from autolabel_modular.ingest.timesync import load_timesync_table
from autolabel_modular.post.postprocess import (
    apply_conf_filter,
    apply_per_image_top_k,
    apply_per_image_top_k_per_camera,
    nms_3d,
)
from autolabel_modular.settings.config import AppConfig, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT
from autolabel_modular.settings.dino_camera_config import camera_top_k_map


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class FrameEntry:
    frame_idx: int
    timestamp_ns: Optional[int]
    pre_objs: List[Dict[str, Any]]   # NMS 后、未做 temporal 的框（重新跑一遍得到）
    post_objs: List[Dict[str, Any]]  # annotations_demo.json 里的 final 框


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _corners_of(obj: Dict[str, Any]) -> Optional[np.ndarray]:
    b = obj.get("bbox_3d_lidar") or {}
    c = b.get("corners")
    if c is None:
        return None
    arr = np.asarray(c, dtype=np.float64)
    if arr.shape != (8, 3):
        return None
    return arr


def _palette(i: int) -> Tuple[int, int, int]:
    base = [
        (0, 140, 255), (0, 220, 0), (255, 80, 80), (200, 120, 255),
        (0, 220, 220), (255, 200, 0), (180, 60, 180), (60, 200, 180),
        (255, 120, 255), (120, 255, 120), (80, 80, 255), (200, 200, 40),
        (255, 180, 80), (60, 255, 255), (180, 180, 255), (255, 60, 120),
    ]
    if i < 0:
        return (140, 140, 140)  # 未分配 track 的灰色
    return base[i % len(base)]


def _obb_rect_in_bev(corners: np.ndarray) -> np.ndarray:
    """从 8 个角点估计底面矩形（按主轴 PCA 再做 AABB），返回 4x2（xy）。"""
    xy = corners[:, :2]
    c = xy.mean(axis=0)
    diffs = xy - c
    cov = np.cov(diffs.T)
    if np.all(np.isfinite(cov)):
        evals, evecs = np.linalg.eigh(cov)
        main = evecs[:, int(np.argmax(evals))]
        yaw = float(np.arctan2(main[1], main[0]))
    else:
        yaw = 0.0
    ca, sa = math.cos(yaw), math.sin(yaw)
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    local = diffs @ R
    lo = local.min(axis=0)
    hi = local.max(axis=0)
    rect_local = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return (rect_local @ R.T) + c


# ---------------------------------------------------------------------------
# pre temporal：重跑与 runner 一致的 conf + top-k + nms
# ---------------------------------------------------------------------------


def rebuild_pre_temporal(
    raw_kept: List[List[Dict[str, Any]]],
    *,
    conf_thr: float,
    max_boxes_per_image: int,
    nms_iou_thr: float,
    nms_center_dist_m: float,
    nms_size_overlap_min: float,
    dino_use_camera_specific: bool,
) -> List[List[Dict[str, Any]]]:
    filtered = apply_conf_filter(raw_kept, conf_thr=conf_thr)
    if dino_use_camera_specific:
        after_topk = apply_per_image_top_k_per_camera(filtered, camera_top_k_map(), default_max=10_000)
    else:
        after_topk = apply_per_image_top_k(filtered, max_boxes_per_image=max_boxes_per_image)
    return nms_3d(
        after_topk,
        iou_thr=nms_iou_thr,
        center_near_m=nms_center_dist_m,
        size_overlap_min=nms_size_overlap_min,
    )


# ---------------------------------------------------------------------------
# BEV 画布
# ---------------------------------------------------------------------------


def _world_to_px(x: float, y: float, size: int, half: float) -> Tuple[int, int]:
    scale = 0.5 * size / max(half, 1e-6)
    u = int(round(size * 0.5 - y * scale))
    v = int(round(size * 0.5 - x * scale))
    return u, v


def _init_bev(size: int, half: float, title: str) -> np.ndarray:
    canvas = np.full((size, size, 3), 28, dtype=np.uint8)
    for m in range(-int(half), int(half) + 1, 10):
        u0, v0 = _world_to_px(float(-half), float(m), size, half)
        u1, v1 = _world_to_px(float(half), float(m), size, half)
        cv2.line(canvas, (u0, v0), (u1, v1), (58, 58, 58), 1)
        u0, v0 = _world_to_px(float(m), float(-half), size, half)
        u1, v1 = _world_to_px(float(m), float(half), size, half)
        cv2.line(canvas, (u0, v0), (u1, v1), (58, 58, 58), 1)
    ou, ov = _world_to_px(0.0, 0.0, size, half)
    cv2.drawMarker(canvas, (ou, ov), (210, 210, 210), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    return canvas


def _draw_obj_on_bev(
    canvas: np.ndarray,
    obj: Dict[str, Any],
    size: int,
    half: float,
    *,
    color_override: Optional[Tuple[int, int, int]] = None,
    with_label: bool = True,
) -> None:
    corners = _corners_of(obj)
    if corners is None:
        return
    rect = _obb_rect_in_bev(corners)
    pts = np.array([_world_to_px(float(p[0]), float(p[1]), size, half) for p in rect], dtype=np.int32)
    tid = int(obj.get("temporal_track_id", -1)) if obj.get("temporal_track_id") is not None else -1
    color = color_override if color_override is not None else _palette(tid)
    smoothed = bool(obj.get("temporal_smoothed", False))
    thick = 2 if smoothed else 1
    cv2.polylines(canvas, [pts], True, color, thick)
    c_xy = rect.mean(axis=0)
    cxpx, cypx = _world_to_px(float(c_xy[0]), float(c_xy[1]), size, half)
    cv2.circle(canvas, (cxpx, cypx), 3, color, -1)
    if with_label:
        lab = f"t{tid}" if tid >= 0 else "-"
        cv2.putText(canvas, lab, (cxpx + 5, cypx - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 每帧左右对比：pre vs post
# ---------------------------------------------------------------------------


def render_frame_pair(
    pre_objs: List[Dict[str, Any]],
    post_objs: List[Dict[str, Any]],
    frame_idx: int,
    *,
    size: int = 640,
    half: float = 60.0,
) -> np.ndarray:
    left = _init_bev(size, half, f"BEFORE temporal  (frame {frame_idx})")
    right = _init_bev(size, half, f"AFTER  temporal  (frame {frame_idx})")
    # pre：没有 track_id，用统一灰白色
    for o in pre_objs:
        _draw_obj_on_bev(left, o, size, half,
                        color_override=(210, 210, 210), with_label=False)
    # post：按 track_id 上色
    n_long = 0
    for o in post_objs:
        _draw_obj_on_bev(right, o, size, half)
        if o.get("temporal_refined"):
            n_long += 1
    # 标注统计
    cv2.putText(right, f"long-track boxes: {n_long}/{len(post_objs)}",
                (12, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 1, cv2.LINE_AA)
    cv2.putText(left, f"boxes: {len(pre_objs)}",
                (12, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    combo = np.concatenate([left, right], axis=1)
    cv2.line(combo, (size, 0), (size, size), (80, 80, 80), 1)
    return combo


# ---------------------------------------------------------------------------
# 全序列轨迹图（world 或 LiDAR）
# ---------------------------------------------------------------------------


def render_trajectories(
    per_frame_post_world: List[List[Dict[str, Any]]],
    *,
    size: int = 900,
    half: float = 120.0,
    coord_name: str = "world",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    canvas = _init_bev(size, half, f"All-frame trajectories ({coord_name})")
    # 收集每个 track 的中心轨迹
    track_to_pts: Dict[int, List[Tuple[float, float, int]]] = {}
    for f_idx, objs in enumerate(per_frame_post_world):
        for o in objs:
            tid = int(o.get("temporal_track_id", -1)) if o.get("temporal_track_id") is not None else -1
            corners = _corners_of(o)
            if corners is None:
                continue
            c = corners[:, :2].mean(axis=0)
            track_to_pts.setdefault(tid, []).append((float(c[0]), float(c[1]), f_idx))

    long_cnt = 0
    total_cnt = len(track_to_pts) - (1 if -1 in track_to_pts else 0)
    for tid, pts in track_to_pts.items():
        if tid < 0:
            # 未关联上的，画小灰点
            for x, y, _ in pts:
                u, v = _world_to_px(x, y, size, half)
                cv2.circle(canvas, (u, v), 2, (130, 130, 130), -1)
            continue
        color = _palette(tid)
        is_long = len(pts) >= 3
        if is_long:
            long_cnt += 1
        # 画连续的折线
        px_pts = [_world_to_px(x, y, size, half) for x, y, _ in sorted(pts, key=lambda t: t[2])]
        for i in range(1, len(px_pts)):
            cv2.line(canvas, px_pts[i - 1], px_pts[i], color, 2 if is_long else 1, cv2.LINE_AA)
        # 起点/终点
        cv2.circle(canvas, px_pts[0], 4, color, -1)
        cv2.circle(canvas, px_pts[-1], 4, color, 1)
        if is_long:
            u, v = px_pts[-1]
            cv2.putText(canvas, f"t{tid}", (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"tracks={total_cnt}, long(>=3)={long_cnt}",
                (12, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

    stats = {
        "total_tracks": int(total_cnt),
        "long_tracks": int(long_cnt),
        "n_frames": len(per_frame_post_world),
    }
    return canvas, stats


# ---------------------------------------------------------------------------
# HTML 汇总
# ---------------------------------------------------------------------------


def write_html(
    out_dir: Path,
    frame_thumbs: List[Tuple[int, str]],
    traj_name: str,
    pipeline_stats: Dict[str, Any],
    traj_stats: Dict[str, Any],
) -> Path:
    lines = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Temporal Refinement Effect</title>",
        "<style>"
        "body{background:#111;color:#eee;font-family:system-ui;margin:20px}"
        "h2{color:#fff} .muted{color:#aaa} .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}"
        ".grid a{color:#9cf;text-decoration:none;border:1px solid #333;padding:4px;background:#1b1b1b}"
        ".grid img{width:100%;display:block;border-radius:4px}"
        "table{border-collapse:collapse;margin:10px 0} td,th{border:1px solid #333;padding:4px 10px}"
        "img.big{max-width:100%;border:1px solid #333;border-radius:6px}"
        "</style>",
        "<h2>Temporal Refinement — 效果可视化</h2>",
        "<p class='muted'>左=未做 temporal（仅 conf+Top-K+3D NMS），右=最终（+temporal±ego）。不同颜色表示不同 <code>temporal_track_id</code>，粗线=被平滑的 long track。</p>",
        "<h3>关键统计</h3>",
        "<table>",
    ]
    for k in ["ego_pose_used", "track_count", "long_track_count", "short_track_count",
              "refinement_count", "duplicate_reduced", "objects_before", "objects_after",
              "iou_threshold", "window_frames", "smoothing_method"]:
        v = pipeline_stats.get(k, "-")
        lines.append(f"<tr><td><code>{k}</code></td><td>{html.escape(str(v))}</td></tr>")
    for k, v in traj_stats.items():
        lines.append(f"<tr><td><code>traj.{k}</code></td><td>{html.escape(str(v))}</td></tr>")
    lines.append("</table>")
    lines.append(f"<h3>全序列轨迹图</h3><img class='big' src='{traj_name}' loading='lazy'/>")
    lines.append("<h3>每帧 BEV 对比（左=前，右=后）</h3><div class='grid'>")
    for f_idx, name in frame_thumbs:
        lines.append(f"<a href='{name}' target='_blank'><img src='{name}' loading='lazy'/><div>frame {f_idx}</div></a>")
    lines.append("</div>")
    out = out_dir / "tracks_overview.html"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 主 run
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
    out_dir = output_root / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_root / "annotations_demo_raw.json"
    final_path = output_root / "annotations_demo.json"
    stats_path = output_root / "stage_stats.json"
    if not raw_path.is_file() or not final_path.is_file():
        print(f"[错误] 需要先跑 pipeline；找不到 {raw_path} 或 {final_path}", file=sys.stderr)
        return 2

    raw_doc = _load_json(raw_path)
    final_doc = _load_json(final_path)
    stats_doc = _load_json(stats_path) if stats_path.is_file() else {}
    temporal_stats: Dict[str, Any] = stats_doc.get("temporal_stats") or {}

    # 从 metadata 读取上次跑的 dino 模式（保持 pre vs post 一致）
    meta = final_doc.get("metadata") or {}
    cfg_saved = (meta.get("config") or {})
    dino_use_camera_specific = bool(cfg_saved.get("dino_use_camera_specific", True))

    # 从 raw_kept 重跑 conf+topk+nms → pre-temporal 状态
    raw_frames = raw_doc.get("frames") or []
    final_frames = final_doc.get("frames") or []
    n_raw = len(raw_frames)
    n_final = len(final_frames)
    if n_raw != n_final:
        print(f"[警告] raw_kept frames({n_raw}) 与 final frames({n_final}) 数量不一致，取较小值")
    n_frames = min(n_raw, n_final)
    if args.max_frames and args.max_frames > 0:
        n_frames = min(n_frames, int(args.max_frames))

    pre_per_frame = [list(fr.get("objects") or []) for fr in raw_frames[:n_frames]]
    post_per_frame = [list(fr.get("objects") or []) for fr in final_frames[:n_frames]]

    pre_nms = rebuild_pre_temporal(
        pre_per_frame,
        conf_thr=float(args.conf_thr),
        max_boxes_per_image=int(args.max_boxes_per_image),
        nms_iou_thr=float(args.nms_iou_thr),
        nms_center_dist_m=float(args.nms_center_dist_m),
        nms_size_overlap_min=float(args.nms_size_overlap_min),
        dino_use_camera_specific=dino_use_camera_specific,
    )

    # ego-pose（可选）：用来把 post 搬到 world 系画轨迹
    use_ego = bool(args.ego_pose) and bool(temporal_stats.get("ego_pose_used", False))
    Ts_world_lidar: List[Optional[np.ndarray]] = [None] * n_frames
    post_world: List[List[Dict[str, Any]]] = post_per_frame
    coord_name = "LiDAR (frame0)"
    if use_ego:
        try:
            cfg = AppConfig(data_root=data_root, output_root=output_root, max_frames=n_frames)
            frames = load_timesync_table(cfg.data_root, cfg.max_frames)[:n_frames]
            calib = load_and_validate_calibration(cfg.data_root / "calibration.json", cfg.cameras)
            cache = EgoPoseCache.from_dir(cfg.data_root / cfg.ego_pose_dir_name)
            Ts_world_lidar, _rep = build_per_frame_T_world_lidar(
                cache, frames, vTl=calib.vTl,
                max_dt_ns=int(cfg.ego_pose_max_dt_ns),
                use_first_frame_as_origin=bool(cfg.ego_pose_use_first_frame_origin),
                angle_in_degrees=bool(cfg.ego_pose_angle_in_degrees),
            )
            post_world, _ = transform_boxes_to_world(post_per_frame, Ts_world_lidar)
            coord_name = "world (ego-compensated)"
        except Exception as e:
            print(f"[警告] ego 变换失败，轨迹图回退到 LiDAR(frame0) 系：{type(e).__name__}: {e}")
            post_world = post_per_frame
            coord_name = "LiDAR (frame0, ego-fallback)"

    # 逐帧渲染 BEV 对比
    thumbs: List[Tuple[int, str]] = []
    for f_idx in range(n_frames):
        img = render_frame_pair(
            pre_nms[f_idx], post_per_frame[f_idx], f_idx,
            size=int(args.bev_size), half=float(args.bev_half_m),
        )
        name = f"bev_frame_{f_idx:06d}.jpg"
        cv2.imwrite(str(out_dir / name), img)
        thumbs.append((f_idx, name))

    # 全序列轨迹图
    traj_img, traj_stats = render_trajectories(
        post_world, size=int(args.traj_size), half=float(args.traj_half_m), coord_name=coord_name,
    )
    traj_name = "trajectories_bev.jpg"
    cv2.imwrite(str(out_dir / traj_name), traj_img)

    html_path = write_html(out_dir, thumbs, traj_name, temporal_stats, traj_stats)

    # 写一个小 run_meta.json 方便审计
    (out_dir / "run_meta.json").write_text(
        json.dumps({
            "n_frames": n_frames,
            "use_ego": bool(use_ego),
            "coord_name": coord_name,
            "bev_half_m": float(args.bev_half_m),
            "traj_half_m": float(args.traj_half_m),
            "temporal_stats": temporal_stats,
            "traj_stats": traj_stats,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] 输出目录: {out_dir}")
    print(f"[OK] 帧对比图: {len(thumbs)} 张")
    print(f"[OK] 轨迹图:   {out_dir / traj_name}")
    print(f"[OK] HTML:     {html_path}")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Temporal Refinement effect visualization")
    ap.add_argument("--data-root", type=str, default="")
    ap.add_argument("--output-root", type=str, default="")
    ap.add_argument("--output-subdir", type=str, default="temporal_effect")
    ap.add_argument("--max-frames", type=int, default=0, help="0 表示用全部帧")
    # pre-temporal 重建参数（与 cli/main.py 默认一致）
    ap.add_argument("--conf-thr", type=float, default=0.18)
    ap.add_argument("--max-boxes-per-image", type=int, default=8)
    ap.add_argument("--nms-iou-thr", type=float, default=0.3)
    ap.add_argument("--nms-center-dist-m", type=float, default=2.0)
    ap.add_argument("--nms-size-overlap-min", type=float, default=0.3)
    # ego 开关
    ap.add_argument("--ego-pose", action="store_true",
                    help="尝试把 post 框变到 world 系后再画轨迹（需要 annotations 来自 --ego-pose 的跑）")
    # BEV 尺寸
    ap.add_argument("--bev-size", type=int, default=640)
    ap.add_argument("--bev-half-m", type=float, default=60.0)
    ap.add_argument("--traj-size", type=int, default=900)
    ap.add_argument("--traj-half-m", type=float, default=120.0)
    return ap.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
