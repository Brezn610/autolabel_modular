from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..calib.calibration import (
    build_distortion_coeffs,
    get_T_lidar_to_cam,
    get_image_size_hw,
    load_and_validate_calibration,
    parse_intrinsic_to_K,
)
from ..detect.sam_refinement import apply_sam_mask_to_frustum, draw_sam_frustum_debug_bgr
from ..detect.detection import (
    grounding_dino_detect,
    load_grounding_dino,
    map_label_to_class,
    take_top_k_by_score,
)
from ..export.exporter import build_annotations_export, write_json
from ..geom.projection import frustum_mask_uvz, project_lidar_to_image
from ..ingest.lidar_io import downsample_points, load_lidar_xyz
from ..ingest.timesync import load_timesync_table
from ..lift3d.lifting import cluster_and_fit_obb, obb_to_json_dict
from ..post.postprocess import (
    apply_conf_filter,
    apply_per_image_top_k,
    apply_per_image_top_k_per_camera,
    collect_scores,
    nms_3d,
    summarize_scores,
)
from ..settings.config import AppConfig
from ..settings.dino_camera_config import DINO_CAMERA_CONFIG, camera_top_k_map, dino_prompt_and_top_k
from ..temporal.temporal_refinement import draw_temporal_debug, refine_tracklets
from ..ego_pose.ego_pose import (
    EgoPoseCache,
    build_per_frame_T_world_lidar,
    transform_boxes_from_world,
    transform_boxes_to_world,
)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe_value(v) for v in value]
    return value


def _colorize_by_depth(z: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    zn = (z - zmin) / max(zmax - zmin, 1e-6)
    zn = np.clip(zn, 0.0, 1.0)
    c = (zn * 255).astype(np.uint8)
    return cv2.applyColorMap(c, cv2.COLORMAP_JET).reshape(-1, 3)


def _draw_projection_overlay(
    image_bgr: np.ndarray,
    uv: np.ndarray,
    z_cam: np.ndarray,
    in_front_mask: np.ndarray,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    canvas = image_bgr.copy()
    valid = in_front_mask & np.isfinite(uv).all(axis=1)
    uu = np.round(uv[valid, 0]).astype(np.int32)
    vv = np.round(uv[valid, 1]).astype(np.int32)
    inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    uu, vv = uu[inside], vv[inside]
    zv = z_cam[valid][inside]
    if zv.size == 0:
        return canvas
    colors = _colorize_by_depth(zv, float(np.percentile(zv, 5)), float(np.percentile(zv, 95)))
    for (u, v), c in zip(zip(uu, vv), colors):
        cv2.circle(canvas, (int(u), int(v)), 1, (int(c[0]), int(c[1]), int(c[2])), -1)
    return canvas


def _run_projection_visualization(cfg: AppConfig, calib: Dict[str, Any], frames: List[Any]) -> Path:
    out_dir = cfg.output_root / "projection_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fr in tqdm(frames, desc="投影可视化"):
        lidar_name = fr.files["middle_lidar"]
        xyz_full = load_lidar_xyz(cfg.data_root, lidar_name)
        xyz = downsample_points(xyz_full, cfg.max_points_projection)
        for cam in cfg.cameras:
            if cam not in fr.files:
                continue
            img_path = cfg.data_root / cam / fr.files[cam]
            if not img_path.is_file():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            T_l2c = get_T_lidar_to_cam(calib, cam)
            intr = calib[cam]["intrinsics"]
            K = parse_intrinsic_to_K(intr)
            dist = build_distortion_coeffs(intr)
            h, w = get_image_size_hw(intr)
            if img.shape[:2] != (h, w):
                pass
            uv, zc = project_lidar_to_image(xyz, T_l2c, K, dist)
            vis = _draw_projection_overlay(img, uv, zc, in_front_mask=(zc > 1e-3))
            out_name = f"projection_vis_{fr.frame_index:03d}__{cam}.jpg"
            cv2.imwrite(str(out_dir / out_name), vis)
    return out_dir


def run_pipeline(
    cfg: AppConfig,
    conf_thr: float,
    max_boxes_per_image: int,
    nms_iou_thr: float,
    nms_center_dist_m: float = 2.0,
    nms_size_overlap_min: float = 0.3,
    save_projection_vis: bool = False,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    calib_bundle = load_and_validate_calibration(cfg.data_root / "calibration.json", cfg.cameras)
    calib = calib_bundle.raw
    frames = load_timesync_table(cfg.data_root, cfg.max_frames)
    projection_vis_dir = ""
    if save_projection_vis:
        projection_vis_dir = str(_run_projection_visualization(cfg=cfg, calib=calib, frames=frames))
    processor, gdino_model = load_grounding_dino(cfg.grounding_dino_model_id, device)

    sam_use_any = bool(cfg.sam2_debug_enabled or cfg.sam_enabled_in_frustum)
    sam2_processor = sam2_model = None
    if sam_use_any:
        from ..detect.sam2_masks import (
            draw_sam_overlay_bgr,
            load_sam2,
            mask_to_zlib_b64,
            sam2_masks_from_boxes,
        )

        print(f"[SAM2] 加载模型 {cfg.sam2_model_id} …")
        sam2_processor, sam2_model = load_sam2(cfg.sam2_model_id, device)

    sam_debug_dir = cfg.output_root / "sam_debug"
    if sam_use_any:
        sam_debug_dir.mkdir(parents=True, exist_ok=True)

    per_frame_objects: List[List[Dict[str, Any]]] = [[] for _ in frames]
    raw_dino_debug: List[Dict[str, Any]] = []
    stage_stats = {"raw_dets": 0, "frustum_nonempty": 0, "obb_success": 0}
    sam_frustum_stats: Dict[str, Any] = {
        "enabled": bool(cfg.sam_enabled_in_frustum),
        "rows": 0,
        "dets_with_sam_array": 0,
        "mask_intersection_ok": 0,
        "fallback_to_box_count": 0,
        "sum_frustum_points_box_only": 0,
        "sum_frustum_points_final": 0,
    }
    vis_rng = np.random.default_rng(0)

    class_names = list(cfg.class_colors_bgr.keys())

    for fr in tqdm(frames, desc="检测+3D提升"):
        lidar_name = fr.files["middle_lidar"]
        xyz_full = load_lidar_xyz(cfg.data_root, lidar_name)
        frame_debug = {"frame_idx": fr.frame_index, "cameras": {}}

        for cam in cfg.cameras_for_detection:
            if cam not in fr.files:
                continue
            img_path = cfg.data_root / cam / fr.files[cam]
            if not img_path.is_file():
                continue

            pil = Image.open(img_path).convert("RGB")
            if cfg.dino_use_camera_specific:
                cam_prompt, cam_top_k = dino_prompt_and_top_k(cam)
                dets = grounding_dino_detect(
                    processor=processor,
                    model=gdino_model,
                    pil_image=pil,
                    prompt=cam_prompt,
                    box_thr=cfg.dino_recall_box_threshold,
                    text_thr=cfg.dino_recall_text_threshold,
                    device=device,
                )
                dets = take_top_k_by_score(dets, cam_top_k)
            else:
                dets = grounding_dino_detect(
                    processor=processor,
                    model=gdino_model,
                    pil_image=pil,
                    prompt=cfg.dino_text_prompt,
                    box_thr=cfg.dino_box_threshold,
                    text_thr=cfg.dino_text_threshold,
                    device=device,
                )
            stage_stats["raw_dets"] += len(dets)

            T_l2c = get_T_lidar_to_cam(calib, cam)
            intr = calib[cam]["intrinsics"]
            K = parse_intrinsic_to_K(intr)
            dist = build_distortion_coeffs(intr)
            uv, zc = project_lidar_to_image(xyz_full, T_l2c, K, dist)
            calib_h, calib_w = get_image_size_hw(intr)

            boxes_for_sam = [list(det["box_xyxy"]) for det in dets]
            if sam_use_any and sam2_processor is not None and boxes_for_sam:
                sam_list = sam2_masks_from_boxes(
                    sam2_processor,
                    sam2_model,
                    pil,
                    boxes_for_sam,
                    device,
                    mask_threshold=cfg.sam2_mask_threshold,
                )
            else:
                sam_list = [{} for _ in dets]

            cam_debug = {"num_dets": len(dets), "detections": []}
            uv_draw_per_det: List[np.ndarray] = []
            for det, sam_extra in zip(dets, sam_list):
                sx = sam_extra or {}
                m_hw = sx.get("sam_mask_hw")
                if isinstance(m_hw, np.ndarray) and m_hw.size > 0:
                    det["sam_mask_hw"] = m_hw
                else:
                    det.pop("sam_mask_hw", None)

                cls = map_label_to_class(det["label"], class_names)
                box = det["box_xyxy"]
                frustum_sub: Dict[str, Any] = {}
                if cfg.sam_enabled_in_frustum:
                    sam_frustum_stats["rows"] += 1
                    mask, frustum_sub = apply_sam_mask_to_frustum(
                        uv,
                        zc,
                        box,
                        cfg.frustum_z_min,
                        cfg.frustum_z_max,
                        m_hw,
                        (calib_h, calib_w),
                        dilate_iters=cfg.sam_frustum_mask_dilate_iters,
                        min_points=cfg.sam_frustum_fallback_min_points,
                    )
                    sm = m_hw if isinstance(m_hw, np.ndarray) and m_hw.size > 0 else None
                    if sm is not None:
                        sam_frustum_stats["dets_with_sam_array"] += 1
                    if frustum_sub.get("used_sam_mask"):
                        if frustum_sub.get("fallback_to_box"):
                            sam_frustum_stats["fallback_to_box_count"] += 1
                        else:
                            sam_frustum_stats["mask_intersection_ok"] += 1
                    sam_frustum_stats["sum_frustum_points_box_only"] += int(
                        frustum_sub.get("n_box_only", 0)
                    )
                    sam_frustum_stats["sum_frustum_points_final"] += int(frustum_sub.get("n_final", 0))
                else:
                    mask = frustum_mask_uvz(uv, zc, box, cfg.frustum_z_min, cfg.frustum_z_max)
                pts = xyz_full[mask]
                frustum_points = int(pts.shape[0])
                if frustum_points > 0:
                    stage_stats["frustum_nonempty"] += 1

                uv_sel = uv[mask]
                if uv_sel.shape[0] > cfg.sam_debug_max_points_draw:
                    pick = vis_rng.choice(
                        uv_sel.shape[0], size=cfg.sam_debug_max_points_draw, replace=False
                    )
                    uv_sel = uv_sel[pick]
                uv_plot = uv_sel.astype(np.float64)
                pw, ph = pil.size[0], pil.size[1]
                if calib_w > 0 and calib_h > 0 and (pw != calib_w or ph != calib_h):
                    uv_plot[:, 0] *= pw / float(calib_w)
                    uv_plot[:, 1] *= ph / float(calib_h)
                uv_draw_per_det.append(uv_plot)

                obb = cluster_and_fit_obb(
                    pts.astype("float64"),
                    cfg.dbscan_eps,
                    cfg.dbscan_min_samples,
                    yaw_only=cfg.obb_yaw_only,
                )
                obb_ok = obb is not None
                if obb_ok:
                    stage_stats["obb_success"] += 1
                    obj_json = obb_to_json_dict(obb, cls, det["score"])
                    obj_json["bbox_2d"] = {"camera": cam, "xyxy": box, "dino_label": det["label"]}
                    per_frame_objects[fr.frame_index].append(obj_json)

                row: Dict[str, Any] = {
                    "label": det["label"],
                    "score": det["score"],
                    "box_xyxy": box,
                    "frustum_points": frustum_points,
                    "obb_success": obb_ok,
                }
                if cfg.sam_enabled_in_frustum:
                    row["sam_frustum_used_mask"] = bool(frustum_sub.get("used_sam_mask", False))
                    row["sam_frustum_fallback"] = bool(frustum_sub.get("fallback_to_box", False))
                    row["frustum_points_box_only"] = int(frustum_sub.get("n_box_only", 0))
                if sam_use_any:
                    row["sam_mask_area_px"] = int(sx.get("sam_mask_area_px", 0))
                    if sx.get("sam_best_iou") is not None:
                        row["sam_best_iou"] = sx["sam_best_iou"]
                    if "sam_mask_variant_idx" in sx:
                        row["sam_mask_variant_idx"] = sx["sam_mask_variant_idx"]
                    if sx.get("sam_error"):
                        row["sam_error"] = sx["sam_error"]
                    if cfg.sam2_debug_store_compressed_mask and sx.get("sam_mask_hw") is not None:
                        b64, shp = mask_to_zlib_b64(sx["sam_mask_hw"])
                        row["sam_mask_shape_hw"] = shp
                        row["sam_mask_zlib_b64"] = b64
                cam_debug["detections"].append(row)

            if sam_use_any and sam2_processor is not None and dets:
                bgr = cv2.imread(str(img_path))
                if bgr is not None:
                    if bgr.shape[1] != pil.size[0] or bgr.shape[0] != pil.size[1]:
                        bgr = cv2.resize(bgr, (pil.size[0], pil.size[1]))
                    if cfg.sam_enabled_in_frustum:
                        overlay = draw_sam_frustum_debug_bgr(bgr, boxes_for_sam, sam_list, uv_draw_per_det)
                    else:
                        overlay = draw_sam_overlay_bgr(bgr, boxes_for_sam, sam_list)
                    cv2.imwrite(
                        str(sam_debug_dir / f"frame_{fr.frame_index:06d}__{cam}.jpg"),
                        overlay,
                    )

            frame_debug["cameras"][cam] = cam_debug
        raw_dino_debug.append(frame_debug)

    raw_kept = copy.deepcopy(per_frame_objects)
    score_summary_raw_kept = summarize_scores(collect_scores(raw_kept))
    filtered = apply_conf_filter(per_frame_objects, conf_thr=conf_thr)
    if cfg.dino_use_camera_specific:
        after_topk = apply_per_image_top_k_per_camera(filtered, camera_top_k_map(), default_max=10_000)
    else:
        after_topk = apply_per_image_top_k(filtered, max_boxes_per_image=max_boxes_per_image)
    after_nms = nms_3d(
        after_topk,
        iou_thr=nms_iou_thr,
        center_near_m=nms_center_dist_m,
        size_overlap_min=nms_size_overlap_min,
    )

    temporal_stats: Dict[str, Any] = {"enabled": bool(cfg.temporal_enabled)}
    temporal_debug_dir = ""
    final_objects: List[List[Dict[str, Any]]] = after_nms
    if cfg.temporal_enabled:
        try:
            # ---- 可选：ego-pose 先把 OBB 变到 world 系，让 BEV IoU 关联更稳定 ----
            ego_used = False
            ego_report: Dict[str, Any] = {"enabled": bool(cfg.ego_pose_enabled)}
            Ts_world_lidar: List[Optional[np.ndarray]] = [None] * len(after_nms)
            objects_for_temporal: List[List[Dict[str, Any]]] = after_nms
            if cfg.ego_pose_enabled:
                try:
                    ego_dir = cfg.data_root / cfg.ego_pose_dir_name
                    ego_cache = EgoPoseCache.from_dir(ego_dir)
                    if not ego_cache.timestamps:
                        raise RuntimeError(f"ego 缓存为空，目录: {ego_dir}")
                    Ts_world_lidar, ego_report_build = build_per_frame_T_world_lidar(
                        ego_cache, frames, vTl=calib_bundle.vTl,
                        max_dt_ns=int(cfg.ego_pose_max_dt_ns),
                        use_first_frame_as_origin=bool(cfg.ego_pose_use_first_frame_origin),
                        angle_in_degrees=bool(cfg.ego_pose_angle_in_degrees),
                    )
                    objects_for_temporal, ego_to_stats = transform_boxes_to_world(
                        after_nms, Ts_world_lidar
                    )
                    ego_report.update({
                        "ok": True,
                        **ego_report_build,
                        "transform_to_world": ego_to_stats,
                    })
                    ego_used = True
                except Exception as e:
                    ego_report.update({
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    Ts_world_lidar = [None] * len(after_nms)
                    objects_for_temporal = after_nms
            temporal_stats["ego_pose"] = ego_report
            temporal_stats["ego_pose_used"] = bool(ego_used)
            temporal_stats["ego_pose_fallback_count"] = int(
                ego_report.get("n_missing", 0)
            ) if ego_used else int(len(frames) if cfg.ego_pose_enabled else 0)

            refined, t_stats = refine_tracklets(
                objects_for_temporal,
                iou_threshold=cfg.temporal_iou_threshold,
                window_frames=cfg.temporal_window_frames,
                min_dets_per_track=cfg.temporal_min_dets_per_track,
                smoothing_method=cfg.temporal_smoothing_method,
                max_center_gate_m=cfg.temporal_center_gate_m,
            )

            # 如果之前把 box 送到了 world 系，现在要把 refined 结果转回 LiDAR 系
            # 好让导出/可视化/已有下游逻辑都保持一致
            if ego_used:
                refined_back, ego_back_stats = transform_boxes_from_world(refined, Ts_world_lidar)
                temporal_stats["ego_pose"]["transform_from_world"] = ego_back_stats
                final_objects = refined_back
            else:
                final_objects = refined

            temporal_stats.update(t_stats)
            temporal_stats["fallback_used"] = False
            if cfg.temporal_debug_vis:
                try:
                    # 用 world 系下的 refined 画 BEV（更能看出 track 是否稳定）
                    td_src = refined if ego_used else final_objects
                    td_path = draw_temporal_debug(td_src, cfg.output_root / "temporal_debug")
                    temporal_debug_dir = str(td_path)
                except Exception as e:
                    temporal_stats["debug_vis_error"] = f"{type(e).__name__}: {e}"
        except Exception as e:
            # 安全回退：任何异常都退回到 3D NMS 的结果
            final_objects = after_nms
            temporal_stats["fallback_used"] = True
            temporal_stats["error"] = f"{type(e).__name__}: {e}"

    # 把 temporal_track_id 反写到 raw_dino_debug 的每条 detection row
    # 采用 (camera, box_xyxy) 做精确匹配：final_objects 里保留了 bbox_2d.camera 和 xyxy
    if cfg.temporal_enabled and not temporal_stats.get("fallback_used", True):
        # 按 frame_idx 建索引：(cam, tuple(xyxy)) -> (track_id, smoothed, refined)
        for fr, objs in zip(frames, final_objects):
            lookup: Dict[Any, Dict[str, Any]] = {}
            for o in objs:
                b2 = o.get("bbox_2d") or {}
                cam = b2.get("camera")
                xyxy = b2.get("xyxy")
                if cam is None or xyxy is None:
                    continue
                key = (str(cam), tuple(float(v) for v in xyxy))
                lookup[key] = {
                    "temporal_track_id": o.get("temporal_track_id", -1),
                    "temporal_smoothed": bool(o.get("temporal_smoothed", False)),
                    "temporal_refined": bool(o.get("temporal_refined", False)),
                }
            # 找到对应 frame 在 raw_dino_debug 中的位置
            fd = None
            for x in raw_dino_debug:
                if x.get("frame_idx") == fr.frame_index:
                    fd = x
                    break
            if fd is None:
                continue
            for cam_name, cam_block in (fd.get("cameras") or {}).items():
                for row in cam_block.get("detections", []):
                    key = (str(cam_name), tuple(float(v) for v in (row.get("box_xyxy") or [])))
                    if key in lookup:
                        row.update(lookup[key])

    metadata = {
        "dataset_root": str(cfg.data_root),
        "num_frames": len(frames),
        "cameras_detection": cfg.cameras_for_detection,
        "grounding_dino_model": cfg.grounding_dino_model_id,
        "prompt": (
            cfg.dino_text_prompt
            if not cfg.dino_use_camera_specific
            else "per-camera (see dino_camera_config)"
        ),
        "dino_mode": "camera_specific_topk" if cfg.dino_use_camera_specific else "global_threshold",
        "dino_camera_config": _json_safe_value(DINO_CAMERA_CONFIG) if cfg.dino_use_camera_specific else None,
        "dino_recall_thresholds": (
            {"box": cfg.dino_recall_box_threshold, "text": cfg.dino_recall_text_threshold}
            if cfg.dino_use_camera_specific
            else None
        ),
        "config": _json_safe_value(asdict(cfg)),
        "sam2_debug": cfg.sam2_debug_enabled,
        "sam_enabled_in_frustum": cfg.sam_enabled_in_frustum,
        "sam2_model_id": cfg.sam2_model_id if sam_use_any else None,
        "temporal_enabled": cfg.temporal_enabled,
        "temporal": {
            "window_frames": cfg.temporal_window_frames,
            "iou_threshold": cfg.temporal_iou_threshold,
            "smoothing_method": cfg.temporal_smoothing_method,
            "min_dets_per_track": cfg.temporal_min_dets_per_track,
            "center_gate_m": cfg.temporal_center_gate_m,
        } if cfg.temporal_enabled else None,
        "ego_pose_enabled": cfg.ego_pose_enabled,
        "ego_pose": {
            "dir_name": cfg.ego_pose_dir_name,
            "max_dt_ns": cfg.ego_pose_max_dt_ns,
            "use_first_frame_origin": cfg.ego_pose_use_first_frame_origin,
        } if cfg.ego_pose_enabled else None,
    }

    out_raw_ann = cfg.output_root / "annotations_demo_raw.json"
    out_final_ann = cfg.output_root / "annotations_demo.json"
    out_raw_debug = cfg.output_root / "raw_dino_debug.json"
    out_stats = cfg.output_root / "stage_stats.json"

    if cfg.sam_enabled_in_frustum:
        r = max(1, int(sam_frustum_stats.get("rows", 0)))
        sam_frustum_stats["mean_points_box_only"] = float(sam_frustum_stats["sum_frustum_points_box_only"]) / r
        sam_frustum_stats["mean_points_final"] = float(sam_frustum_stats["sum_frustum_points_final"]) / r

    write_json(out_raw_ann, build_annotations_export(frames, raw_kept, metadata))
    write_json(out_final_ann, build_annotations_export(frames, final_objects, metadata))
    debug_doc: Dict[str, Any] = {"frames": raw_dino_debug}
    if sam_use_any:
        debug_doc["sam2_meta"] = {
            "model_id": cfg.sam2_model_id,
            "mask_threshold": cfg.sam2_mask_threshold,
            "store_compressed_mask": cfg.sam2_debug_store_compressed_mask,
            "sam_debug_dir": str(sam_debug_dir),
        }
    if cfg.sam_enabled_in_frustum:
        debug_doc["sam_frustum_meta"] = {
            "fallback_min_points": cfg.sam_frustum_fallback_min_points,
            "mask_dilate_iters": cfg.sam_frustum_mask_dilate_iters,
            "max_points_draw": cfg.sam_debug_max_points_draw,
        }
    if cfg.temporal_enabled:
        debug_doc["temporal_meta"] = {
            "window_frames": cfg.temporal_window_frames,
            "iou_threshold": cfg.temporal_iou_threshold,
            "smoothing_method": cfg.temporal_smoothing_method,
            "min_dets_per_track": cfg.temporal_min_dets_per_track,
            "center_gate_m": cfg.temporal_center_gate_m,
            "stats": temporal_stats,
            "debug_dir": temporal_debug_dir,
        }
    write_json(out_raw_debug, debug_doc)

    stats = {
        "stage_counts": stage_stats,
        "sam_frustum_stats": sam_frustum_stats,
        "temporal_stats": temporal_stats,
        "raw_kept_score_summary": score_summary_raw_kept,
        "paths": {
            "annotations_raw": str(out_raw_ann),
            "annotations_final": str(out_final_ann),
            "raw_dino_debug": str(out_raw_debug),
            "projection_vis_dir": projection_vis_dir,
            "sam_debug_dir": str(sam_debug_dir) if sam_use_any else "",
            "temporal_debug_dir": temporal_debug_dir,
        },
    }
    write_json(out_stats, stats)
    return stats
