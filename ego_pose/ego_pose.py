"""Ego-pose 读取与变换工具。

数据来源（DrivIng `day/`）
--------------------------------
- ``vehicle_state/*.json``：10Hz，单文件一帧，字段有
    timestamp_nanoseconds, pos_rel_x, pos_rel_y, height_msl, roll, pitch, yaw
- ``calibration.json``：``middle_lidar.extrinsics.vTl`` 是 4x4 T_vehicle_from_lidar

约定
------------
- 我们构造“序列局部世界系（sequence-local world）”：把序列首帧的 ego 位置视为 world 原点，
  高度取该帧 ``height_msl`` 作为 world 的 z=0，避免 height_msl 绝对值过大带来的数值问题。
- yaw/pitch/roll 按车辆常用的 *ZYX intrinsic* 约定装配（yaw 先绕 z、再 pitch 绕 y、再 roll 绕 x）。
  DrivIng 实际轴规范通过静止目标在世界系中的“是否稳定”做闭环验证。
- 一切函数都设计成 **安全回退**：找不到 pose、时间戳缺失、数值异常时返回 4x4 单位阵并写入 error，
  由调用方决定是否回退到 sensor coord。

对外接口
----------
- :class:`EgoPoseCache`：读入整个 ``vehicle_state/`` 目录，按 timestamp 索引，支持最近邻 O(logN) 查询。
- :func:`load_ego_pose`：外部便捷接口，给定 timestamp_ns 返回 ego dict（已带 dt_ns 诊断项）。
- :func:`build_T_world_ego`：从 ego dict 装配 4x4 T_world_ego。
- :func:`transform_boxes_to_world` / :func:`transform_boxes_from_world`：批量把 3D OBB 在 LiDAR 系
  与 world 系之间来回搬。
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 旋转矩阵装配（ZYX intrinsic：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)）
# ---------------------------------------------------------------------------


def _R_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _yaw_from_R(R: np.ndarray) -> float:
    # 仅取 z 轴分量（temporal refinement 只关心 yaw）
    return float(math.atan2(float(R[1, 0]), float(R[0, 0])))


# ---------------------------------------------------------------------------
# EgoPoseCache：扫描 vehicle_state/ 建立 timestamp → 文件索引
# ---------------------------------------------------------------------------


@dataclass
class EgoPoseCache:
    """把 ``vehicle_state/*.json`` 按 timestamp_nanoseconds 建成一个可最近邻查询的缓存。

    实现要点（为 WSL/远程盘上 2w+ 小 JSON 优化）
    ------------------------------------------------
    - **只扫文件名**（DrivIng 的文件名本身就是 ns 时间戳）而不是全部读 JSON；
    - 真正要用到的帧才 **按需懒加载** 对应 JSON；
    - 同时缓存首帧 ``origin``，避免每次 world 变换都重新读一次文件。

    典型用法：
        >>> cache = EgoPoseCache.from_dir(Path('/mnt/d/thesis/day/vehicle_state'))
        >>> ego = cache.nearest(1750163715800028000)
        >>> T, meta = build_T_world_ego(ego, origin_anchor=cache.origin)
    """

    timestamps: List[int]
    paths: List[Path]
    origin: Optional[Dict[str, float]] = None
    source_dir: Optional[Path] = None
    _cache: Dict[int, Dict[str, Any]] = None  # lazy parsed entries keyed by index

    @classmethod
    def from_dir(cls, directory: Path, *, include_sweeps_dir: Optional[Path] = None) -> "EgoPoseCache":
        entries: List[Tuple[int, Path]] = []

        def _scan(d: Optional[Path]) -> None:
            if d is None or not d.is_dir():
                return
            for p in d.iterdir():
                if p.suffix != ".json":
                    continue
                try:
                    ts_int = int(p.stem)
                except Exception:
                    continue
                entries.append((ts_int, p))

        _scan(directory)
        _scan(include_sweeps_dir)
        entries.sort(key=lambda x: x[0])
        timestamps = [e[0] for e in entries]
        paths = [e[1] for e in entries]

        origin: Optional[Dict[str, float]] = None
        if paths:
            try:
                e0 = json.loads(paths[0].read_text(encoding="utf-8"))
                origin = {
                    "pos_rel_x": float(e0.get("pos_rel_x", 0.0)),
                    "pos_rel_y": float(e0.get("pos_rel_y", 0.0)),
                    "height_msl": float(e0.get("height_msl", 0.0)),
                    "yaw": float(e0.get("yaw", 0.0)),
                }
            except Exception:
                origin = None

        return cls(
            timestamps=timestamps,
            paths=paths,
            origin=origin,
            source_dir=directory,
            _cache={},
        )

    def _read(self, index: int) -> Optional[Dict[str, Any]]:
        if self._cache is None:
            self._cache = {}
        if index in self._cache:
            return self._cache[index]
        try:
            obj = json.loads(self.paths[index].read_text(encoding="utf-8"))
        except Exception:
            return None
        self._cache[index] = obj
        return obj

    def nearest(self, timestamp_ns: int) -> Optional[Dict[str, Any]]:
        """返回最接近 ``timestamp_ns`` 的 ego dict（附加 ``_dt_ns`` 字段）。"""
        if not self.timestamps:
            return None
        ts = int(timestamp_ns)
        i = bisect.bisect_left(self.timestamps, ts)
        candidates: List[int] = []
        if i < len(self.timestamps):
            candidates.append(i)
        if i > 0:
            candidates.append(i - 1)
        if not candidates:
            return None
        best = min(candidates, key=lambda k: abs(self.timestamps[k] - ts))
        obj = self._read(best)
        if obj is None:
            return None
        entry = dict(obj)
        entry["_dt_ns"] = int(self.timestamps[best] - ts)
        return entry


def load_ego_pose(
    cache: EgoPoseCache,
    timestamp_ns: Optional[int],
    *,
    max_dt_ns: int = 60_000_000,  # 60ms 兜底（10Hz 周期 100ms），超出视为找不到
) -> Optional[Dict[str, Any]]:
    """按时间戳取最近邻的 ego_state。时间对不上（或偏差 > ``max_dt_ns``）则返回 None。"""
    if timestamp_ns is None:
        return None
    try:
        ts = int(timestamp_ns)
    except Exception:
        return None
    entry = cache.nearest(ts)
    if entry is None:
        return None
    if abs(int(entry.get("_dt_ns", 0))) > int(max_dt_ns):
        return None
    return entry


# ---------------------------------------------------------------------------
# 4x4 T_world_ego
# ---------------------------------------------------------------------------


def build_T_world_ego(
    ego: Optional[Dict[str, Any]],
    *,
    origin_anchor: Optional[Dict[str, float]] = None,
    angle_in_degrees: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """用 (roll, pitch, yaw, pos_rel_x, pos_rel_y, height_msl) 装 4x4 T_world_ego。

    注：DrivIng `vehicle_state/*.json` 的 roll/pitch/yaw 为 **度**（经数据验证 yaw∈[0,360]、
    roll/pitch 为小角度），默认 ``angle_in_degrees=True`` 做一次 deg→rad。

    若以 ``origin_anchor``（通常是序列首帧 ego 字段）为 world 原点，则平移先减去该原点。
    返回 (T 4x4, meta)。异常场景返回单位阵 + ``ok=False``。
    """
    meta: Dict[str, Any] = {"ok": False, "reason": None}
    if ego is None:
        meta["reason"] = "ego_is_none"
        return np.eye(4, dtype=np.float64), meta
    try:
        roll = float(ego.get("roll", 0.0))
        pitch = float(ego.get("pitch", 0.0))
        yaw = float(ego.get("yaw", 0.0))
        tx = float(ego.get("pos_rel_x", 0.0))
        ty = float(ego.get("pos_rel_y", 0.0))
        tz = float(ego.get("height_msl", 0.0))
    except Exception as e:  # pragma: no cover
        meta["reason"] = f"parse_error:{type(e).__name__}:{e}"
        return np.eye(4, dtype=np.float64), meta

    if angle_in_degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

    if origin_anchor is not None:
        tx -= float(origin_anchor.get("pos_rel_x", 0.0))
        ty -= float(origin_anchor.get("pos_rel_y", 0.0))
        tz -= float(origin_anchor.get("height_msl", 0.0))

    R = _R_from_rpy(roll, pitch, yaw)
    if not np.all(np.isfinite(R)):
        meta["reason"] = "non_finite_rotation"
        return np.eye(4, dtype=np.float64), meta
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    meta.update({"ok": True, "dt_ns": int(ego.get("_dt_ns", 0)), "rpy": [roll, pitch, yaw]})
    return T, meta


# ---------------------------------------------------------------------------
# OBB 变换：lidar_frame ↔ world
# ---------------------------------------------------------------------------


def _build_R_yaw(yaw: float) -> np.ndarray:
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _corners_from_cev_yaw(center: Sequence[float], extent: Sequence[float], yaw: float) -> np.ndarray:
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


def _transform_one(obj: Dict[str, Any], T_dst_src: np.ndarray) -> Optional[Dict[str, Any]]:
    b = obj.get("bbox_3d_lidar") or {}
    center = b.get("center")
    extent = b.get("extent")
    yaw = b.get("yaw_rad")
    if center is None or extent is None or yaw is None:
        return None
    try:
        c = np.asarray(center, dtype=np.float64).reshape(3)
        e = np.asarray(extent, dtype=np.float64).reshape(3)
        y = float(yaw)
    except Exception:
        return None

    R_dst_src = np.asarray(T_dst_src, dtype=np.float64)[:3, :3]
    t_dst_src = np.asarray(T_dst_src, dtype=np.float64)[:3, 3]
    new_center = R_dst_src @ c + t_dst_src

    # yaw 合成：本地 yaw = src 的 z 旋转；dst 里的 yaw = Rz(T_dst_src) + 本地 yaw
    dyaw = _yaw_from_R(R_dst_src)
    new_yaw = float(y + dyaw)

    new_R = _build_R_yaw(new_yaw)
    new_corners = _corners_from_cev_yaw(new_center, e, new_yaw)

    new_obj = dict(obj)
    new_b = dict(b)
    new_b["center"] = new_center.tolist()
    new_b["extent"] = e.tolist()
    new_b["yaw_rad"] = new_yaw
    new_b["R"] = new_R.tolist()
    new_b["corners"] = new_corners.tolist()
    new_obj["bbox_3d_lidar"] = new_b
    return new_obj


def transform_boxes_to_world(
    per_frame_objects: List[List[Dict[str, Any]]],
    T_world_lidar_per_frame: List[Optional[np.ndarray]],
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, int]]:
    """把每帧的 3D box 从 LiDAR 系变到 world 系。

    对应帧若 T 为 None（找不到 ego），则原样保留并在 stats.fallback 中累加，
    该帧的 obj 不会被 Temporal Refinement 影响（关联失败而已，不会崩）。
    """
    assert len(per_frame_objects) == len(T_world_lidar_per_frame)
    out: List[List[Dict[str, Any]]] = []
    stats = {"n_frames": 0, "n_frames_ok": 0, "n_frames_fallback": 0, "n_obj_transformed": 0, "n_obj_kept_as_is": 0}
    for f, (objs, T) in enumerate(zip(per_frame_objects, T_world_lidar_per_frame)):
        stats["n_frames"] += 1
        if T is None or not np.all(np.isfinite(T)):
            stats["n_frames_fallback"] += 1
            out.append([dict(o) for o in objs])
            stats["n_obj_kept_as_is"] += len(objs)
            continue
        stats["n_frames_ok"] += 1
        row: List[Dict[str, Any]] = []
        for o in objs:
            no = _transform_one(o, T)
            if no is None:
                row.append(dict(o))  # 无 corners 等字段，原样保留
                stats["n_obj_kept_as_is"] += 1
            else:
                row.append(no)
                stats["n_obj_transformed"] += 1
        out.append(row)
    return out, stats


def transform_boxes_from_world(
    per_frame_objects: List[List[Dict[str, Any]]],
    T_world_lidar_per_frame: List[Optional[np.ndarray]],
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, int]]:
    """world → lidar 的反向变换。当 T 缺失或奇异时回退为原样保留。"""
    assert len(per_frame_objects) == len(T_world_lidar_per_frame)
    out: List[List[Dict[str, Any]]] = []
    stats = {"n_frames": 0, "n_frames_ok": 0, "n_frames_fallback": 0, "n_obj_transformed": 0, "n_obj_kept_as_is": 0}
    for f, (objs, T) in enumerate(zip(per_frame_objects, T_world_lidar_per_frame)):
        stats["n_frames"] += 1
        if T is None or not np.all(np.isfinite(T)):
            stats["n_frames_fallback"] += 1
            out.append([dict(o) for o in objs])
            stats["n_obj_kept_as_is"] += len(objs)
            continue
        try:
            Tinv = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            stats["n_frames_fallback"] += 1
            out.append([dict(o) for o in objs])
            stats["n_obj_kept_as_is"] += len(objs)
            continue
        stats["n_frames_ok"] += 1
        row: List[Dict[str, Any]] = []
        for o in objs:
            no = _transform_one(o, Tinv)
            if no is None:
                row.append(dict(o))
                stats["n_obj_kept_as_is"] += 1
            else:
                row.append(no)
                stats["n_obj_transformed"] += 1
        out.append(row)
    return out, stats


# ---------------------------------------------------------------------------
# 一站式：per-frame T_world_lidar
# ---------------------------------------------------------------------------


def build_per_frame_T_world_lidar(
    cache: EgoPoseCache,
    frames: Sequence[Any],
    vTl: np.ndarray,
    *,
    max_dt_ns: int = 60_000_000,
    use_first_frame_as_origin: bool = True,
    angle_in_degrees: bool = True,
) -> Tuple[List[Optional[np.ndarray]], Dict[str, Any]]:
    """对 ``frames`` 里每一帧，查询最近的 ego_state 并装 T_world_lidar = T_world_ego @ vTl。

    - ``frames`` 元素需至少有 ``timestamp_ns``（FrameRecord 已有）；
    - ``vTl`` 是 4x4 T_vehicle_from_lidar；
    - ``use_first_frame_as_origin=True`` 时把首帧 ego 的位置作为 world 原点，提高数值稳定性。

    返回 (Ts, report)。Ts 长度等于 frames 长度；未命中帧为 None。
    """
    vTl_np = np.asarray(vTl, dtype=np.float64)
    if vTl_np.shape != (4, 4):
        raise ValueError(f"vTl 必须是 4x4，当前 {vTl_np.shape}")

    origin_anchor: Optional[Dict[str, float]] = None
    if use_first_frame_as_origin and cache.origin is not None:
        origin_anchor = dict(cache.origin)

    Ts: List[Optional[np.ndarray]] = []
    n_ok = 0
    n_missing = 0
    dt_abs_list: List[int] = []
    for fr in frames:
        ts = getattr(fr, "timestamp_ns", None)
        try:
            ts_int = int(ts) if ts is not None else None
        except Exception:
            ts_int = None
        ego = load_ego_pose(cache, ts_int, max_dt_ns=max_dt_ns)
        if ego is None:
            Ts.append(None)
            n_missing += 1
            continue
        T_we, meta = build_T_world_ego(ego, origin_anchor=origin_anchor, angle_in_degrees=angle_in_degrees)
        if not meta.get("ok", False):
            Ts.append(None)
            n_missing += 1
            continue
        T_wl = T_we @ vTl_np
        Ts.append(T_wl)
        n_ok += 1
        dt_abs_list.append(abs(int(ego.get("_dt_ns", 0))))

    report = {
        "n_frames": int(len(frames)),
        "n_ok": int(n_ok),
        "n_missing": int(n_missing),
        "max_dt_ns_used": int(max_dt_ns),
        "mean_dt_ns": int(float(np.mean(dt_abs_list))) if dt_abs_list else 0,
        "p90_dt_ns": int(float(np.percentile(dt_abs_list, 90))) if dt_abs_list else 0,
        "origin_anchor": origin_anchor,
        "ego_cache_size": int(len(cache.timestamps)),
    }
    return Ts, report
