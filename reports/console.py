from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_detections(raw_debug_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for fr in raw_debug_doc.get("frames", []):
        frame_idx = int(fr.get("frame_idx", -1))
        cameras = fr.get("cameras", {})
        for cam, cam_info in cameras.items():
            for det in cam_info.get("detections", []):
                row = dict(det)
                row["frame_idx"] = frame_idx
                row["camera"] = str(cam)
                flat.append(row)
    return flat


def _score_summary(scores: List[float]) -> str:
    if not scores:
        return "N=0"
    arr = np.asarray(scores, dtype=np.float64)
    return (
        f"N={arr.size}, min={np.min(arr):.4f}, p10={np.percentile(arr,10):.4f}, "
        f"median={np.median(arr):.4f}, p90={np.percentile(arr,90):.4f}, "
        f"max={np.max(arr):.4f}, mean={np.mean(arr):.4f}"
    )


def print_report(raw_dino_debug_path: Path, stage_stats_path: Path) -> None:
    raw_doc = _load_json(raw_dino_debug_path)
    stats_doc = _load_json(stage_stats_path)
    flat = _flatten_detections(raw_doc)

    all_scores = [float(x.get("score", 0.0)) for x in flat]
    frustum_nonempty = [x for x in flat if int(x.get("frustum_points", 0)) > 0]
    obb_success = [x for x in flat if bool(x.get("obb_success", False))]

    print("\n==================== 诊断报告（分段） ====================")
    print("\n[2D 检测阶段：Grounding DINO]")
    print(f"- 原始检测总数: {len(flat)}")
    print(f"- raw score 分布: {_score_summary(all_scores)}")

    print("\n[投影一致性阶段：2D box -> frustum 点]")
    print(f"- frustum_nonempty 数: {len(frustum_nonempty)}")
    print(f"- 通过比例: {len(frustum_nonempty)}/{len(flat)} = {_safe_ratio(len(frustum_nonempty), len(flat)):.2%}")
    low_frustum = [x for x in flat if int(x.get("frustum_points", 0)) <= 3]
    print(f"- frustum 点数<=3 的检测数: {len(low_frustum)} ({_safe_ratio(len(low_frustum), len(flat)):.2%})")

    print("\n[3D 提升阶段：DBSCAN + OBB]")
    print(f"- OBB 成功数: {len(obb_success)}")
    print(f"- 相对 raw det 成功率: {len(obb_success)}/{len(flat)} = {_safe_ratio(len(obb_success), len(flat)):.2%}")
    print(
        f"- 相对 frustum_nonempty 成功率: "
        f"{len(obb_success)}/{len(frustum_nonempty)} = {_safe_ratio(len(obb_success), len(frustum_nonempty)):.2%}"
    )

    sam_meta = raw_doc.get("sam2_meta") or {}
    sam_ious: List[float] = []
    for x in flat:
        v = x.get("sam_best_iou")
        if isinstance(v, (int, float)):
            sam_ious.append(float(v))
    sam_frustum = bool(stats_doc.get("sam_frustum_stats", {}).get("enabled"))
    if sam_meta or sam_ious:
        title = "[SAM2 mask + 视锥取点]" if sam_frustum else "[SAM2 诊断 mask（仅可视化 / 可选 zlib）]"
        print(f"\n{title}")
        print(f"- 模型: {sam_meta.get('model_id', '（见 raw_dino_debug.sam2_meta）')}")
        print(f"- 可视化目录: {sam_meta.get('sam_debug_dir', '（未记录）')}")
        if sam_ious:
            print(f"- sam_best_iou 分布: {_score_summary(sam_ious)}")
        n_err = sum(1 for x in flat if x.get("sam_error"))
        if n_err:
            print(f"- SAM 推理异常条目数: {n_err}")
        areas = [int(x.get("sam_mask_area_px", 0)) for x in flat if int(x.get("sam_mask_area_px", 0)) > 0]
        if areas:
            arr = np.asarray(areas, dtype=np.float64)
            print(
                f"- sam_mask_area_px: N={arr.size}, median={np.median(arr):.0f}, "
                f"p90={np.percentile(arr, 90):.0f}"
            )

    by_cam: Dict[str, List[Dict[str, Any]]] = {}
    for x in flat:
        by_cam.setdefault(str(x["camera"]), []).append(x)

    print("\n[按相机分解]")
    for cam in sorted(by_cam.keys()):
        items = by_cam[cam]
        n = len(items)
        n_fr = sum(int(i.get("frustum_points", 0)) > 0 for i in items)
        n_obb = sum(bool(i.get("obb_success", False)) for i in items)
        scores = [float(i.get("score", 0.0)) for i in items]
        print(
            f"- {cam}: det={n}, frustum_pass={_safe_ratio(n_fr,n):.2%}, "
            f"obb_pass={_safe_ratio(n_obb,n):.2%}, score_median={np.median(scores):.4f}" if n > 0
            else f"- {cam}: det=0"
        )

    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for x in flat:
        by_frame.setdefault(int(x["frame_idx"]), []).append(x)
    frame_rank = []
    for fi, items in by_frame.items():
        n = len(items)
        n_low = sum(int(i.get("frustum_points", 0)) <= 3 for i in items)
        frame_rank.append((fi, n, n_low, _safe_ratio(n_low, n)))
    frame_rank.sort(key=lambda t: t[3], reverse=True)

    print("\n[问题帧 Top-5：frustum 点数低占比最高]")
    for fi, n, n_low, ratio in frame_rank[:5]:
        print(f"- frame={fi}: low_frustum={n_low}/{n} ({ratio:.2%})")

    stage_counts = stats_doc.get("stage_counts", {})
    print("\n[阶段总计（stage_stats.json）]")
    print(f"- raw_dets={stage_counts.get('raw_dets', 0)}")
    print(f"- frustum_nonempty={stage_counts.get('frustum_nonempty', 0)}")
    print(f"- obb_success={stage_counts.get('obb_success', 0)}")
    sfs = stats_doc.get("sam_frustum_stats") or {}
    if sfs.get("enabled"):
        print("\n[SAM 视锥统计 sam_frustum_stats]")
        print(f"- dets_with_sam_array={sfs.get('dets_with_sam_array', 0)}")
        print(f"- mask_intersection_ok={sfs.get('mask_intersection_ok', 0)}")
        print(f"- fallback_to_box_count={sfs.get('fallback_to_box_count', 0)}")
        print(f"- sum_frustum_points_box_only={sfs.get('sum_frustum_points_box_only', 0)}")
        print(f"- sum_frustum_points_final={sfs.get('sum_frustum_points_final', 0)}")

    tstats = stats_doc.get("temporal_stats") or {}
    if tstats.get("enabled"):
        print("\n[Temporal Refinement 统计 temporal_stats（参考 MS3D）]")
        if tstats.get("fallback_used"):
            print(f"- 已回退 (fallback_used=True)，原因: {tstats.get('error', '（未记录）')}")
        else:
            ep = tstats.get("ego_pose") or {}
            if ep.get("enabled"):
                if tstats.get("ego_pose_used") and ep.get("ok"):
                    print(
                        f"- Ego-pose: ENABLED ok，n_ok={ep.get('n_ok', 0)}/"
                        f"{ep.get('n_frames', 0)}，missing={ep.get('n_missing', 0)}，"
                        f"mean|dt|={ep.get('mean_dt_ns', 0)/1e6:.2f}ms, p90={ep.get('p90_dt_ns', 0)/1e6:.2f}ms"
                    )
                else:
                    print(f"- Ego-pose: ENABLED 但回退，原因: {ep.get('error','（未记录）')}")
            print(f"- 关联方法: Hungarian {'(scipy)' if tstats.get('has_scipy_hungarian') else '(greedy)'}")
            print(f"- 平滑方法: {tstats.get('smoothing_method', '-')}")
            print(
                f"- track_count={tstats.get('track_count', 0)} "
                f"(long={tstats.get('long_track_count', 0)}, short={tstats.get('short_track_count', 0)})"
            )
            print(f"- refinement_count={tstats.get('refinement_count', 0)}（被平滑替换的 obj 数）")
            print(
                f"- duplicate_reduced={tstats.get('duplicate_reduced', 0)}"
                f"（同 track 同帧跨相机重复，被合并丢弃的 obj 数）"
            )
            before = int(tstats.get("objects_before", 0))
            after = int(tstats.get("objects_after", 0))
            if before > 0:
                print(f"- 3D 对象数 before→after: {before} → {after} (Δ={after-before})")
    print("=========================================================\n")
