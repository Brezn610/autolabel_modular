"""Temporal Refinement（时序精炼）核心实现。

设计动机
--------
MS3D (darrenjkt/MS3D, T-IV & ITSC 2024) 的主要贡献之一是 *Temporal Refinement*：
    - 跨帧关联（Tracklet Association）：把不同帧/不同相机检出的同一目标串成一条轨迹；
    - 轨迹平滑（Trajectory Smoothing）：在轨迹上做 moving average / spline 平滑，抑制抖动；
    - 轨迹内去重：每条 tracklet 每帧只保留一条最优 3D box。

与 MS3D 的对应关系（引用）
------------------------------
* MS3D/tracker/tracker.py → 我们的 ``_Track`` 数据结构与 ``_associate_with_tracks`` 二部匹配；
* MS3D/tracker/ab3dmot.py 中的 IoU gating → 我们的 ``_bev_iou`` + ``iou_threshold``；
* MS3D/temporal_refinement.py 中的 "box_smoothing" 子例程 → ``_smooth_track``（MA / spline）。

与 MS3D 的差异（重要）
------------------------------
MS3D 完整算法依赖 **ego-pose**（车辆全局位姿）把每帧检测统一到世界坐标系后再关联。
本项目当前的 autolabel 流程只提供 LiDAR 传感器坐标系（middle_lidar）下的 OBB，
因此我们在 **LiDAR 传感器坐标系** 下做关联：
    - 对短窗口、低自运动场景（例如 demo ≤ 30 帧）效果良好；
    - 若 ego 位移明显、或有 odometry，可把每帧的 OBB 先变换到世界系后再送入本模块。

为了稳健，我们采用 BEV AABB IoU（即把 8 个 OBB 角点投到 xy 取 min/max 后再求 IoU）：
这与 ``post.postprocess.nms_3d`` 的度量一致，避免引入 shapely 等新依赖。

对外接口
--------
- :func:`refine_tracklets` 输入 *per_frame_objects*（nms_3d 之后的结果），
  输出相同形状的 *refined_per_frame_objects* 与 *stats*。
  每个 obj 会被打上 ``temporal_track_id``/``temporal_smoothed``/``temporal_refined``。
- :func:`draw_temporal_debug` 为每一帧画 BEV 对比图（不同 track 用不同颜色），
  方便肉眼核对关联与平滑效果。

安全性
--------
整个 ``refine_tracklets`` 在 runner 中的调用点 **要求** 用 try/except 包裹：
任何异常或 track 太短都应 fallback 到原始 nms_3d 的结果，避免影响主流程。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy 在本仓库是默认依赖，但仍做保护
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

try:
    from scipy.interpolate import CubicSpline
    _HAS_SPLINE = True
except Exception:  # pragma: no cover
    _HAS_SPLINE = False


# ---------------------------------------------------------------------------
# 几何辅助（BEV AABB IoU；参考 post/postprocess.py，保持度量一致）
# ---------------------------------------------------------------------------


def _corners_of(obj: Dict[str, Any]) -> Optional[np.ndarray]:
    b = obj.get("bbox_3d_lidar") or {}
    c = b.get("corners")
    if c is None:
        return None
    arr = np.asarray(c, dtype=np.float64)
    if arr.shape != (8, 3):
        return None
    return arr


def _bev_aabb(corners: np.ndarray) -> Tuple[float, float, float, float]:
    """返回 BEV (xmin, ymin, xmax, ymax)。"""
    xy = corners[:, :2]
    return float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())


def _bev_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aw = max(0.0, a[2] - a[0])
    ah = max(0.0, a[3] - a[1])
    bw = max(0.0, b[2] - b[0])
    bh = max(0.0, b[3] - b[1])
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _center_xy(corners: np.ndarray) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = _bev_aabb(corners)
    return 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)


# ---------------------------------------------------------------------------
# Track 数据结构
# ---------------------------------------------------------------------------


@dataclass
class _TrackObs:
    frame_idx: int
    obj_index: int  # obj 在 per_frame_objects[frame_idx] 中的下标
    cls: str
    score: float
    center: Tuple[float, float, float]
    extent: Tuple[float, float, float]
    yaw: float
    bev: Tuple[float, float, float, float]
    camera: str = ""


@dataclass
class _Track:
    track_id: int
    cls: str
    obs: List[_TrackObs] = field(default_factory=list)
    last_frame: int = -1

    @property
    def last_bev(self) -> Tuple[float, float, float, float]:
        return self.obs[-1].bev

    @property
    def last_center(self) -> Tuple[float, float, float]:
        return self.obs[-1].center


# ---------------------------------------------------------------------------
# 单帧 obs 抽取
# ---------------------------------------------------------------------------


def _extract_obs(frame_idx: int, objs: Sequence[Dict[str, Any]]) -> List[_TrackObs]:
    out: List[_TrackObs] = []
    for i, o in enumerate(objs):
        corners = _corners_of(o)
        if corners is None:
            continue
        b3 = o.get("bbox_3d_lidar") or {}
        center = b3.get("center") or [0.0, 0.0, 0.0]
        extent = b3.get("extent") or [1.0, 1.0, 1.0]
        yaw = float(b3.get("yaw_rad", 0.0))
        cam = ((o.get("bbox_2d") or {}).get("camera") or "") if isinstance(o.get("bbox_2d"), dict) else ""
        out.append(
            _TrackObs(
                frame_idx=int(frame_idx),
                obj_index=int(i),
                cls=str(o.get("category", "")),
                score=float(o.get("score", 0.0)),
                center=(float(center[0]), float(center[1]), float(center[2])),
                extent=(float(extent[0]), float(extent[1]), float(extent[2])),
                yaw=yaw,
                bev=_bev_aabb(corners),
                camera=str(cam),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 二部匹配（Hungarian）
# ---------------------------------------------------------------------------


def _associate_with_tracks(
    tracks: List[_Track],
    dets: List[_TrackObs],
    iou_threshold: float,
    max_center_gate_m: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """在 *同类别* 内做 Hungarian 匹配，返回 (matches, unmatched_track_idx, unmatched_det_idx)。"""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    T, D = len(tracks), len(dets)
    BIG = 10.0  # 远大于 1（因为 cost 上限是 1-iou_threshold）
    cost = np.full((T, D), BIG, dtype=np.float64)
    for i, tr in enumerate(tracks):
        tc = tr.last_center
        for j, d in enumerate(dets):
            if tr.cls != d.cls:
                continue
            dx = tc[0] - d.center[0]
            dy = tc[1] - d.center[1]
            if (dx * dx + dy * dy) > (max_center_gate_m * max_center_gate_m):
                continue  # BEV 中心距超门限直接剪枝
            iou = _bev_iou(tr.last_bev, d.bev)
            if iou < iou_threshold:
                continue
            cost[i, j] = 1.0 - iou

    if _HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if cost[r, c] < BIG * 0.5]
    else:  # 贪心回退
        flat = [(cost[r, c], r, c) for r in range(T) for c in range(D) if cost[r, c] < BIG * 0.5]
        flat.sort()
        used_r, used_c = set(), set()
        pairs = []
        for _, r, c in flat:
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))

    matched_r = {p[0] for p in pairs}
    matched_c = {p[1] for p in pairs}
    return pairs, [i for i in range(T) if i not in matched_r], [j for j in range(D) if j not in matched_c]


# ---------------------------------------------------------------------------
# 平滑（moving average / spline），带 yaw unwrap
# ---------------------------------------------------------------------------


def _unwrap_yaws(yaws: Sequence[float]) -> List[float]:
    if not yaws:
        return []
    out = [float(yaws[0])]
    for y in yaws[1:]:
        prev = out[-1]
        d = float(y) - prev
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        out.append(prev + d)
    return out


def _moving_average(values: Sequence[float], window: int = 3) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    k = max(1, int(window))
    pad = k // 2
    a = np.asarray(values, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - pad)
        hi = min(n, i + pad + 1)
        out[i] = float(a[lo:hi].mean())
    return out.tolist()


def _spline_smooth(frames: Sequence[int], values: Sequence[float]) -> List[float]:
    if len(frames) < 4 or not _HAS_SPLINE:
        return _moving_average(values, 3)
    try:
        x = np.asarray(frames, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)
        cs = CubicSpline(x, y, bc_type="natural")
        return [float(cs(f)) for f in x]
    except Exception:
        return _moving_average(values, 3)


def _build_R_yaw(yaw: float) -> np.ndarray:
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _corners_from_cev_yaw(center: Sequence[float], extent: Sequence[float], yaw: float) -> np.ndarray:
    """以 (center, extent, yaw) 重建 8 个角点（与 Open3D OrientedBoundingBox 数学一致）。"""
    R = _build_R_yaw(float(yaw))
    hx, hy, hz = 0.5 * float(extent[0]), 0.5 * float(extent[1]), 0.5 * float(extent[2])
    local = np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
        ],
        dtype=np.float64,
    )
    return (R @ local.T).T + np.asarray(center, dtype=np.float64).reshape(1, 3)


def _smooth_track(
    track: _Track,
    smoothing_method: str,
) -> Dict[str, List[float]]:
    """对一条 track 的 center (x,y,z) + yaw + extent 做平滑，返回每次 obs 对应的平滑结果。"""
    frames = [o.frame_idx for o in track.obs]
    xs = [o.center[0] for o in track.obs]
    ys = [o.center[1] for o in track.obs]
    zs = [o.center[2] for o in track.obs]
    ex = [o.extent[0] for o in track.obs]
    ey = [o.extent[1] for o in track.obs]
    ez = [o.extent[2] for o in track.obs]
    yaws = _unwrap_yaws([o.yaw for o in track.obs])

    if smoothing_method == "spline":
        sx = _spline_smooth(frames, xs)
        sy = _spline_smooth(frames, ys)
        sz = _spline_smooth(frames, zs)
        syaw = _spline_smooth(frames, yaws)
    else:  # moving_average
        sx = _moving_average(xs, 3)
        sy = _moving_average(ys, 3)
        sz = _moving_average(zs, 3)
        syaw = _moving_average(yaws, 3)

    # extent 取轨迹中位数更稳（避免个别帧 DBSCAN 抽风）
    med = [float(np.median(ex)), float(np.median(ey)), float(np.median(ez))]
    sex = [med[0]] * len(frames)
    sey = [med[1]] * len(frames)
    sez = [med[2]] * len(frames)

    return {
        "x": sx, "y": sy, "z": sz,
        "yaw": syaw,
        "ex": sex, "ey": sey, "ez": sez,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def refine_tracklets(
    per_frame_objects: List[List[Dict[str, Any]]],
    *,
    iou_threshold: float = 0.5,
    window_frames: int = 7,
    min_dets_per_track: int = 3,
    smoothing_method: str = "moving_average",
    max_center_gate_m: float = 4.0,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    """跨帧关联 + 轨迹平滑 + 轨迹内去重。

    参数
    ----
    per_frame_objects : 每帧的 3D obj 列表（nms_3d 之后）。
    iou_threshold     : BEV IoU 匹配门限（低于视为不匹配）。
    window_frames     : 允许 track 连续丢失最多多少帧仍可继续关联（>= 1）。
    min_dets_per_track: 轨迹最小长度（小于则不做平滑，仅打 track_id 且 refined=False）。
    smoothing_method  : "moving_average" 或 "spline"。
    max_center_gate_m : BEV 中心距超出该值即剪枝，加速关联。

    返回
    ----
    (refined_per_frame_objects, stats)
        refined_per_frame_objects 与输入同形状；每个 obj 新增：
            - temporal_track_id: int
            - temporal_smoothed: bool
            - temporal_refined:  bool
          被合并（dedup）的重复 obj 不会出现在输出中。
        stats 含 track_count / refinement_count / duplicate_reduced 等。
    """
    n_frames = len(per_frame_objects)
    tracks: List[_Track] = []
    next_id = 0

    # ---- Step 1. 在线关联 ----
    # 备注：Hungarian 强制 1-to-1，本身不会把“同一帧多相机重复检测”挂到同一条 track。
    # 我们用二段式：先 Hungarian 做主匹配，再对未匹配 det 做“本帧已匹配 track 的 IoU 吸附”，
    # 这样跨相机重复就会被合并到同一条 track（Step 2 再按 (track, frame) 只保留最高分）。
    dup_attach_thr = max(iou_threshold, 0.3)  # 二段式用略严一点的 IoU
    active: List[_Track] = []
    for f in range(n_frames):
        dets = _extract_obs(f, per_frame_objects[f])
        pairs, un_tr, un_det = _associate_with_tracks(
            active, dets, iou_threshold=iou_threshold, max_center_gate_m=max_center_gate_m
        )
        # 记录“本帧”哪些 track 刚被匹配，用于把重复 det 吸附过去
        just_matched: Dict[int, _TrackObs] = {}
        for ti, di in pairs:
            active[ti].obs.append(dets[di])
            active[ti].last_frame = f
            just_matched[ti] = dets[di]

        # 二段式：未匹配 det 若与本帧已匹配的 det IoU 较高 → 当作同一 track 的重复
        still_unmatched = []
        for di in un_det:
            d = dets[di]
            best_ti, best_iou = -1, 0.0
            for ti, mobs in just_matched.items():
                if active[ti].cls != d.cls:
                    continue
                iou = _bev_iou(mobs.bev, d.bev)
                if iou >= dup_attach_thr and iou > best_iou:
                    best_iou, best_ti = iou, ti
            if best_ti >= 0:
                active[best_ti].obs.append(d)  # Step 2 的 per-frame-best 会丢弃较低分副本
            else:
                still_unmatched.append(di)

        # 新建未匹配检测 → 新 track
        for di in still_unmatched:
            d = dets[di]
            tr = _Track(track_id=next_id, cls=d.cls, obs=[d], last_frame=f)
            next_id += 1
            active.append(tr)
            tracks.append(tr)

        # 关掉太久未见到的 track
        still_active: List[_Track] = []
        for tr in active:
            if f - tr.last_frame <= max(1, int(window_frames)):
                still_active.append(tr)
        active = still_active

    # ---- Step 2. 平滑 + 构造 refined_per_frame_objects ----
    refined: List[List[Dict[str, Any]]] = [[] for _ in range(n_frames)]
    # 记录：同一 (track, frame) 出现几次（跨相机重复）
    duplicates_per_track_frame = 0
    long_tracks = 0
    short_tracks = 0
    short_track_obs = 0
    smoothed_obs = 0

    for tr in tracks:
        # 同一 track 同一 frame 可能有多条 obs（跨相机），取 score 最高
        per_frame_best: Dict[int, _TrackObs] = {}
        for o in tr.obs:
            prev = per_frame_best.get(o.frame_idx)
            if prev is None or o.score > prev.score:
                if prev is not None:
                    duplicates_per_track_frame += 1
                per_frame_best[o.frame_idx] = o
            else:
                duplicates_per_track_frame += 1

        # 把 obs 按 frame 排序（稳定的 obs 视图用于平滑）
        ordered = sorted(per_frame_best.values(), key=lambda x: x.frame_idx)
        ordered_track = _Track(track_id=tr.track_id, cls=tr.cls, obs=ordered, last_frame=tr.last_frame)
        long_enough = len(ordered) >= max(1, int(min_dets_per_track))

        if long_enough:
            long_tracks += 1
            sm = _smooth_track(ordered_track, smoothing_method=smoothing_method)
            for idx, o in enumerate(ordered):
                src = per_frame_objects[o.frame_idx][o.obj_index]
                new_obj = dict(src)  # 浅拷贝
                b3 = dict(src.get("bbox_3d_lidar") or {})
                center = [sm["x"][idx], sm["y"][idx], sm["z"][idx]]
                extent = [sm["ex"][idx], sm["ey"][idx], sm["ez"][idx]]
                yaw = float(sm["yaw"][idx])
                R = _build_R_yaw(yaw)
                corners = _corners_from_cev_yaw(center, extent, yaw)
                b3["center"] = [float(x) for x in center]
                b3["extent"] = [float(x) for x in extent]
                b3["yaw_rad"] = yaw
                b3["R"] = R.tolist()
                b3["corners"] = corners.tolist()
                new_obj["bbox_3d_lidar"] = b3
                new_obj["temporal_track_id"] = int(tr.track_id)
                new_obj["temporal_smoothed"] = True
                new_obj["temporal_refined"] = True
                refined[o.frame_idx].append(new_obj)
                smoothed_obs += 1
        else:
            short_tracks += 1
            for o in ordered:
                src = per_frame_objects[o.frame_idx][o.obj_index]
                new_obj = dict(src)
                new_obj["temporal_track_id"] = int(tr.track_id)
                new_obj["temporal_smoothed"] = False
                new_obj["temporal_refined"] = False
                refined[o.frame_idx].append(new_obj)
                short_track_obs += 1

    # 补上“没 corners、未进入任何 track”的对象（例如 invalid），原样放回，打未 refine 标记
    # 做一个 O(n) 集合：已被放入 refined 的 (frame, obj_index)
    used = set()
    for tr in tracks:
        for o in tr.obs:
            used.add((o.frame_idx, o.obj_index))
    for f in range(n_frames):
        for i, o in enumerate(per_frame_objects[f]):
            if (f, i) in used:
                continue
            new_obj = dict(o)
            new_obj.setdefault("temporal_track_id", -1)
            new_obj.setdefault("temporal_smoothed", False)
            new_obj.setdefault("temporal_refined", False)
            refined[f].append(new_obj)

    n_before = sum(len(x) for x in per_frame_objects)
    n_after = sum(len(x) for x in refined)

    stats = {
        "track_count": len(tracks),
        "long_track_count": long_tracks,
        "short_track_count": short_tracks,
        "refinement_count": smoothed_obs,          # 被平滑替换的 obj 个数
        "short_track_obs": short_track_obs,        # 短 track 内的 obj 个数（打 track_id，未 refine）
        "duplicate_reduced": duplicates_per_track_frame,  # dedup 删除的重复 obj 个数
        "objects_before": int(n_before),
        "objects_after": int(n_after),
        "iou_threshold": float(iou_threshold),
        "window_frames": int(window_frames),
        "min_dets_per_track": int(min_dets_per_track),
        "smoothing_method": str(smoothing_method),
        "has_scipy_hungarian": bool(_HAS_SCIPY),
        "has_scipy_spline": bool(_HAS_SPLINE),
    }
    return refined, stats


# ---------------------------------------------------------------------------
# 调试可视化（BEV）
# ---------------------------------------------------------------------------


def _palette(i: int) -> Tuple[int, int, int]:
    # 稳定可复现的调色板（BGR）
    base = [
        (0, 140, 255), (0, 220, 0), (255, 80, 80), (200, 120, 255),
        (0, 220, 220), (255, 200, 0), (180, 60, 180), (60, 200, 180),
        (255, 120, 255), (120, 255, 120), (80, 80, 255), (200, 200, 40),
    ]
    return base[i % len(base)]


def draw_temporal_debug(
    refined_per_frame: List[List[Dict[str, Any]]],
    out_dir: Path,
    *,
    view_half_size_m: float = 60.0,
    img_size: int = 720,
) -> Path:
    """把每帧的 BEV（俯视）画出来，按 track_id 上色，保存到 out_dir。

    返回 out_dir（用于统计）。若 cv2 不可用或写入失败，静默降级。
    """
    try:
        import cv2  # 延迟导入
    except Exception:
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    S = int(img_size)
    half = float(view_half_size_m)

    def world_to_px(x: float, y: float) -> Tuple[int, int]:
        # LiDAR x 向前 → 图像向上；LiDAR y 向左 → 图像向左
        # 用常见 BEV 约定：u = S/2 - y*scale，v = S/2 - x*scale
        scale = 0.5 * S / max(half, 1e-6)
        u = int(round(S * 0.5 - y * scale))
        v = int(round(S * 0.5 - x * scale))
        return u, v

    entries: List[Tuple[int, str]] = []
    for f_idx, objs in enumerate(refined_per_frame):
        canvas = np.full((S, S, 3), 30, dtype=np.uint8)
        # 画网格（每 10m 一条）
        for m in range(-int(half), int(half) + 1, 10):
            u0, v0 = world_to_px(float(-half), float(m))
            u1, v1 = world_to_px(float(half), float(m))
            cv2.line(canvas, (u0, v0), (u1, v1), (60, 60, 60), 1)
            u0, v0 = world_to_px(float(m), float(-half))
            u1, v1 = world_to_px(float(m), float(half))
            cv2.line(canvas, (u0, v0), (u1, v1), (60, 60, 60), 1)
        # 原点
        ou, ov = world_to_px(0.0, 0.0)
        cv2.circle(canvas, (ou, ov), 4, (200, 200, 200), -1)
        cv2.putText(canvas, f"frame {f_idx}  (half={half:.0f}m)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)

        for o in objs:
            corners = _corners_of(o)
            if corners is None:
                continue
            tid = int(o.get("temporal_track_id", -1))
            color = _palette(tid if tid >= 0 else 9999)
            # 取底面 4 个角点（z 最小的那 4 个）的凸多边形；若不确定顺序，就用 xy 凸包排序
            xy = corners[:, :2]
            # 简单：取 xy 的 4 个 extreme 构成顺时针矩形（与 yaw 一致）
            # 直接按 BEV AABB 画不够准；改用 yaw 对齐矩形：把 xy 均值作为中心，按 PCA 主轴估 yaw
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
            local = diffs @ R  # 在主轴坐标系
            lo = local.min(axis=0)
            hi = local.max(axis=0)
            rect_local = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            rect_world = (rect_local @ R.T) + c
            pts = np.array([world_to_px(float(p[0]), float(p[1])) for p in rect_world], dtype=np.int32)
            thickness = 2 if o.get("temporal_refined") else 1
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
            # 画中心 + track_id
            cxpx, cypx = world_to_px(float(c[0]), float(c[1]))
            cv2.circle(canvas, (cxpx, cypx), 3, color, -1)
            label = f"t{tid}" if tid >= 0 else "t-"
            cv2.putText(canvas, label, (cxpx + 4, cypx - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        out_path = out_dir / f"bev_frame_{f_idx:06d}.jpg"
        cv2.imwrite(str(out_path), canvas)
        entries.append((f_idx, out_path.name))

    # 简单 index.html
    try:
        html_lines = [
            "<!doctype html><meta charset='utf-8'>",
            "<title>Temporal Refinement BEV Debug</title>",
            "<style>body{background:#111;color:#eee;font-family:system-ui;margin:16px}",
            ".grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}",
            ".grid a{display:block;color:#9cf;text-decoration:none;border:1px solid #333;padding:4px;background:#1b1b1b}",
            ".grid img{width:100%;display:block;border-radius:4px}",
            "</style>",
            "<h2>Temporal Refinement — BEV per frame</h2>",
            "<p>不同颜色 = 不同 track_id；实线粗 = 已平滑 refined，细 = 短 track 原样保留。</p>",
            "<div class='grid'>",
        ]
        for f_idx, name in entries:
            html_lines.append(
                f"<a href='{name}' target='_blank'><img src='{name}' loading='lazy'/><div>frame {f_idx}</div></a>"
            )
        html_lines.append("</div>")
        (out_dir / "index.html").write_text("\n".join(html_lines), encoding="utf-8")
    except Exception:
        pass

    return out_dir
