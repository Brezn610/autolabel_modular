#!/usr/bin/env python3
"""
从 annotations JSON + raw_dino_debug 生成单页 HTML 报告。

要点（与旧版差异）：
- 画布强制对齐标定 ImageSize：磁盘图若与内参尺寸不一致，先缩放再画；2D xyxy 按相同比例缩放，
  使「橙色 3D OBB 投影（K）」与「黄色 DINO 框」在同一像素坐标系下。
- 按 frame_idx 建索引并排序遍历，避免 JSON 帧顺序与 timesync 不一致。
- 生成前清空 report_assets 下本报告使用的 frame_*.jpg，避免残留旧图。
- 页眉写入 demo JSON 的 sha256 前缀与 mtime，便于确认读的是否为最新文件。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from ..calib.calibration import (
    build_distortion_coeffs,
    get_T_lidar_to_cam,
    get_image_size_hw,
    load_and_validate_calibration,
    parse_intrinsic_to_K,
)
from ..ingest.timesync import load_timesync_table


def _iou_xyxy(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _obb_corners_and_edges(b3: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    ctr = b3.get("center")
    Rm = b3.get("R")
    ext = b3.get("extent")
    if ctr is None or Rm is None or ext is None:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=int)
    obb = o3d.geometry.OrientedBoundingBox(
        np.asarray(ctr, dtype=np.float64),
        np.asarray(Rm, dtype=np.float64),
        np.asarray(ext, dtype=np.float64),
    )
    corners = np.asarray(obb.get_box_points(), dtype=np.float64)
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    edges = np.asarray(ls.lines, dtype=np.int64)
    return corners, edges


def _project_points_lidar(
    xyz_lidar: np.ndarray,
    T_l2c: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = xyz_lidar.shape[0]
    pts_h = np.hstack([xyz_lidar, np.ones((n, 1), dtype=np.float64)])
    pts_c = (T_l2c @ pts_h.T).T[:, :3].astype(np.float64)
    pts_cv = pts_c.reshape(-1, 1, 3)
    img_pts, _ = cv2.projectPoints(pts_cv, np.zeros(3), np.zeros(3), K, dist)
    uv = np.asarray(img_pts.reshape(-1, 2), dtype=np.float64)
    z = np.asarray(pts_c[:, 2], dtype=np.float64)
    return uv, z


def _segment_visible(
    uv: np.ndarray,
    z: np.ndarray,
    i: int,
    j: int,
    w: int,
    h: int,
    z_eps: float = 0.05,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if z[i] <= z_eps or z[j] <= z_eps:
        return None
    p1 = (float(uv[i, 0]), float(uv[i, 1]))
    p2 = (float(uv[j, 0]), float(uv[j, 1]))
    if not (np.isfinite(p1[0]) and np.isfinite(p1[1]) and np.isfinite(p2[0]) and np.isfinite(p2[1])):
        return None
    rect = (0, 0, w, h)
    ok, a, b = cv2.clipLine(rect, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))))
    if not ok:
        return None
    return (a, b)


def _sam_banner_html(meta: Dict[str, Any], dbg_doc: Dict[str, Any]) -> str:
    cfg = meta.get("config") or {}
    s2 = dbg_doc.get("sam2_meta") or {}
    sfm = dbg_doc.get("sam_frustum_meta") or {}
    parts: List[str] = []
    if cfg.get("sam_enabled_in_frustum"):
        parts.append("已启用 <b>SAM mask → 视锥取点</b>（与 2D box 求交；点数不足则回退仅用 box）")
    else:
        parts.append("未启用 SAM 视锥（<code>sam_enabled_in_frustum=false</code>；下列 SAM 列可能为「—」）")
    if cfg.get("sam2_debug_enabled"):
        parts.append("已启用 <code>sam2_debug</code>（zlib 等诊断输出）")
    mid = s2.get("model_id")
    if mid:
        parts.append(f'SAM 模型：<code>{html.escape(str(mid))}</code>')
    if sfm:
        fp = sfm.get("fallback_min_points")
        di = sfm.get("mask_dilate_iters")
        parts.append(
            f"视锥元数据：回退阈值 <code>{html.escape(str(fp))}</code> 点、mask 膨胀 <code>{html.escape(str(di))}</code> 次"
        )
    dbg_dir = s2.get("sam_debug_dir")
    if dbg_dir:
        parts.append(f'可视化目录：<code>{html.escape(str(dbg_dir))}</code>')
    return '<p class="muted"><b>SAM 与视锥</b>：' + " <b>|</b> ".join(parts) + "</p>"


def _match_debug_for_object(
    frame_idx: int,
    cam: str,
    box_xyxy: List[float],
    debug_by_frame: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    fr = debug_by_frame.get(frame_idx)
    if not fr:
        return None
    cams = fr.get("cameras") or {}
    ci = cams.get(cam) or {}
    dets = ci.get("detections") or []
    best: Optional[Dict[str, Any]] = None
    best_iou = 0.0
    for d in dets:
        bx = d.get("box_xyxy")
        if not bx or len(bx) != 4:
            continue
        iou = _iou_xyxy([float(x) for x in box_xyxy], [float(x) for x in bx])
        if iou > best_iou:
            best_iou = iou
            best = d
    if best is None or best_iou < 0.05:
        return None
    return best


def _difficulty_score(frustum_points: int, score: float, vol: float, vol_ref: float) -> float:
    p = max(0, int(frustum_points))
    pn = 1.0 - min(1.0, p / 200.0)
    sn = 1.0 - float(np.clip(score, 0.0, 1.0))
    vr = max(vol_ref, 1e-6)
    vn = 1.0 - min(1.0, vol / vr)
    tiny = 1.0 if vol < 0.5 else 0.0
    return float(np.clip(0.45 * pn + 0.35 * sn + 0.15 * vn + 0.05 * tiny, 0.0, 1.0))


def _density(points: int, box_xyxy: List[float]) -> float:
    x1, y1, x2, y2 = box_xyxy
    area = max(1.0, (x2 - x1) * (y2 - y1))
    return float(points) / area * 1e4


def _scale_xyxy(xyxy: List[float], sx: float, sy: float) -> List[float]:
    x1, y1, x2, y2 = [float(t) for t in xyxy]
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def _prepare_canvas_bgr(
    img_bgr: np.ndarray,
    w_calib: int,
    h_calib: int,
) -> Tuple[np.ndarray, float, float]:
    """将磁盘图像缩放到标定尺寸；返回 (canvas, sx, sy) 用于把 JSON 里基于原图尺寸的 xyxy 映射到画布。"""
    h0, w0 = img_bgr.shape[:2]
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid image size")
    canvas = cv2.resize(img_bgr, (w_calib, h_calib), interpolation=cv2.INTER_LINEAR)
    sx = w_calib / float(w0)
    sy = h_calib / float(h0)
    return canvas, sx, sy


def _draw_frame_overlay(
    canvas_bgr: np.ndarray,
    objects: List[Dict[str, Any]],
    cam: str,
    T_l2c: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    xyxy_sx: float,
    xyxy_sy: float,
) -> np.ndarray:
    """在已与标定对齐的画布上绘制：3D OBB 边（橙）+ 缩放后的 2D 框（黄）。"""
    out = canvas_bgr.copy()
    h, w = out.shape[:2]
    for obj in objects:
        b2 = obj.get("bbox_2d") or {}
        if str(b2.get("camera")) != cam:
            continue
        b3 = obj.get("bbox_3d_lidar") or {}
        corners, edge_idx = _obb_corners_and_edges(b3)
        if corners.shape[0] == 8 and edge_idx.size > 0:
            uv, zc = _project_points_lidar(corners, T_l2c, K, dist)
            # OpenCV 为 BGR：(255,128,0) 会被画成偏蓝；橙色应为低 B、高 R
            col_obb = (0, 140, 255)
            for e in edge_idx:
                i, j = int(e[0]), int(e[1])
                seg = _segment_visible(uv, zc, i, j, w, h)
                if seg is None:
                    continue
                cv2.line(out, seg[0], seg[1], col_obb, 2, lineType=cv2.LINE_AA)
        xyxy = b2.get("xyxy")
        if xyxy and len(xyxy) == 4:
            xs = _scale_xyxy(xyxy, xyxy_sx, xyxy_sy)
            x1, y1, x2, y2 = [int(round(float(t))) for t in xs]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 1, lineType=cv2.LINE_AA)
    return out


def _pick_display_camera(objects: List[Dict[str, Any]], preferred: str) -> str:
    cams: List[str] = []
    for o in objects:
        c = str((o.get("bbox_2d") or {}).get("camera") or "")
        if c:
            cams.append(c)
    if not cams:
        return preferred
    if preferred in cams:
        return preferred
    return Counter(cams).most_common(1)[0][0]


def _index_frames_by_idx(frames_list: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], int]:
    out: Dict[int, Dict[str, Any]] = {}
    dup = 0
    for fr in frames_list:
        fi = int(fr.get("frame_idx", -1))
        if fi < 0:
            continue
        if fi in out:
            dup += 1
        out[fi] = fr
    return out, dup


def _file_fingerprint(path: Path) -> Tuple[str, str]:
    """文件大小 + mtime + 前 4MB 内容的哈希（大 JSON 不全盘读）。"""
    if not path.is_file():
        return ("", "")
    st = path.stat()
    mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    h.update(str(st.st_mtime_ns).encode())
    with path.open("rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return (h.hexdigest()[:16], mtime)


def _clear_frame_assets(assets_dir: Path) -> int:
    n = 0
    if not assets_dir.is_dir():
        return 0
    for p in assets_dir.glob("frame_*.jpg"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


def build_html_report(
    data_root: Path,
    demo_json: Path,
    debug_json: Path,
    output_html: Path,
    cameras: List[str],
    display_camera: str,
    difficulty_thr: float,
    max_table_rows: int,
) -> Path:
    if not demo_json.is_file():
        raise FileNotFoundError(f"找不到标注 JSON: {demo_json.resolve()}")
    if not debug_json.is_file():
        raise FileNotFoundError(f"找不到 debug JSON: {debug_json.resolve()}")

    demo_sha, demo_mtime = _file_fingerprint(demo_json)
    dbg_sha, dbg_mtime = _file_fingerprint(debug_json)

    with demo_json.open("r", encoding="utf-8") as f:
        demo = json.load(f)
    with debug_json.open("r", encoding="utf-8") as f:
        dbg_doc = json.load(f)

    meta = demo.get("metadata") or {}
    frames_list: List[Dict[str, Any]] = demo.get("frames") or []
    frames_by_idx, n_dup = _index_frames_by_idx(frames_list)
    sorted_fis = sorted(frames_by_idx.keys())

    debug_by_frame: Dict[int, Dict[str, Any]] = {}
    for fr in dbg_doc.get("frames") or []:
        fi = int(fr.get("frame_idx", -1))
        if fi >= 0:
            debug_by_frame[fi] = fr

    calib_bundle = load_and_validate_calibration(data_root / "calibration.json", cameras)
    calib = calib_bundle.raw

    n_meta = int(meta.get("num_frames", 0))
    max_frames = max(n_meta, len(sorted_fis), (max(sorted_fis) + 1) if sorted_fis else 0)
    ts_frames = load_timesync_table(data_root, max_frames=max_frames)

    assets_dir = output_html.parent / (output_html.stem + "_assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    n_removed = _clear_frame_assets(assets_dir)

    vols: List[float] = []
    for fr in frames_list:
        for o in fr.get("objects") or []:
            b3 = o.get("bbox_3d_lidar") or {}
            ext = b3.get("extent")
            if ext and len(ext) == 3:
                vols.append(float(ext[0] * ext[1] * ext[2]))
    vol_ref = float(np.median(vols)) if vols else 1.0

    cache_bust = datetime.now().strftime("%Y%m%d%H%M%S")
    n_objs_final = sum(len(frames_by_idx[fi].get("objects") or []) for fi in sorted_fis)

    toc_items: List[str] = []
    sections: List[str] = []

    for fi in sorted_fis:
        fr = frames_by_idx[fi]
        ts = fr.get("timestamp_ns")
        objs_all: List[Dict[str, Any]] = list(fr.get("objects") or [])
        cam = _pick_display_camera(objs_all, display_camera)
        objs = [o for o in objs_all if str((o.get("bbox_2d") or {}).get("camera")) == cam]

        toc_items.append(f'<a href="#frame_{fi:06d}">帧 {fi:06d}</a>')

        img_rel = ""
        if fi < len(ts_frames) and cam in ts_frames[fi].files:
            img_path = data_root / cam / ts_frames[fi].files[cam]
            if img_path.is_file():
                img = cv2.imread(str(img_path))
                if img is not None:
                    intr = calib[cam]["intrinsics"]
                    h_calib, w_calib = get_image_size_hw(intr)
                    K = parse_intrinsic_to_K(intr)
                    dist = build_distortion_coeffs(intr)
                    T_l2c = get_T_lidar_to_cam(calib, cam)
                    canvas, sx, sy = _prepare_canvas_bgr(img, w_calib, h_calib)
                    vis = _draw_frame_overlay(canvas, objs, cam, T_l2c, K, dist, sx, sy)
                    out_img = assets_dir / f"frame_{fi:06d}__{cam}.jpg"
                    cv2.imwrite(str(out_img), vis)
                    img_rel = out_img.name

        rows_html: List[str] = []
        difficulties: List[float] = []
        table_objs = sorted(objs, key=lambda o: float(o.get("score", 0.0)), reverse=True)[:max_table_rows]

        sam_iou_cam: List[float] = []
        n_fallback_cam = 0
        for o_sum in objs:
            b2s = o_sum.get("bbox_2d") or {}
            c_s = str(b2s.get("camera", ""))
            xy_s = b2s.get("xyxy") or []
            if len(xy_s) != 4:
                continue
            ms = _match_debug_for_object(fi, c_s, [float(x) for x in xy_s], debug_by_frame)
            if not ms:
                continue
            v = ms.get("sam_best_iou")
            if isinstance(v, (int, float)):
                sam_iou_cam.append(float(v))
            if ms.get("sam_frustum_fallback"):
                n_fallback_cam += 1

        intr_tbl = (calib.get(cam) or {}).get("intrinsics") or {}
        try:
            h_calib_tbl, w_calib_tbl = get_image_size_hw(intr_tbl) if intr_tbl.get("ImageSize") else (0, 0)
        except (KeyError, TypeError, ValueError):
            h_calib_tbl, w_calib_tbl = 0, 0

        for idx, o in enumerate(table_objs, start=1):
            cat = str(o.get("category", "")).upper()
            score = float(o.get("score", 0.0))
            b2 = o.get("bbox_2d") or {}
            cam_o = str(b2.get("camera", ""))
            xyxy = b2.get("xyxy") or [0, 0, 0, 0]
            if len(xyxy) != 4:
                xyxy = [0, 0, 0, 0]

            m = _match_debug_for_object(fi, cam_o, [float(x) for x in xyxy], debug_by_frame)
            frustum_pts = int(m.get("frustum_points", 0)) if m else 0

            b3 = o.get("bbox_3d_lidar") or {}
            ext = b3.get("extent") or [0, 0, 0]
            vol = float(ext[0] * ext[1] * ext[2]) if len(ext) == 3 else 0.0
            dens = _density(frustum_pts, [float(x) for x in xyxy])
            diff = _difficulty_score(frustum_pts, score, vol, vol_ref)
            difficulties.append(diff)

            if diff >= difficulty_thr:
                level = "困难（可能 FP/FN）"
                row_cls = "row-hard"
            else:
                level = "正常"
                row_cls = ""

            sam_iou = m.get("sam_best_iou") if m else None
            iou_cell = f"{float(sam_iou):.3f}" if isinstance(sam_iou, (int, float)) else "—"
            pbox = m.get("frustum_points_box_only") if m else None
            pbox_cell = str(int(pbox)) if isinstance(pbox, (int, float)) else "—"
            used_m = m.get("sam_frustum_used_mask") if m else None
            mask_cell = "是" if used_m is True else ("否" if used_m is False else "—")
            fb_m = m.get("sam_frustum_fallback") if m else None
            fb_cell = "是" if fb_m is True else ("否" if fb_m is False else "—")

            rows_html.append(
                "<tr class='{cls}'>"
                "<td>{idx}</td><td>{cat}</td><td>{score:.3f}</td><td>{diff:.3f}</td>"
                "<td>{level}</td><td>{pts}</td><td>{pbox}</td><td>{iou}</td>"
                "<td>{mask}</td><td>{fb}</td><td>{dens:.4f}</td><td>{vol:.3f}</td><td>{cam}</td>"
                "</tr>".format(
                    cls=row_cls,
                    idx=idx,
                    cat=html.escape(cat),
                    score=score,
                    diff=diff,
                    level=html.escape(level),
                    pts=frustum_pts,
                    pbox=html.escape(pbox_cell),
                    iou=html.escape(iou_cell),
                    mask=html.escape(mask_cell),
                    fb=html.escape(fb_cell),
                    dens=dens,
                    vol=vol,
                    cam=html.escape(cam_o),
                )
            )

        n_hard = sum(1 for d in difficulties if d >= difficulty_thr)
        max_diff = max(difficulties) if difficulties else 0.0
        status = f'<span class="ok">正常</span>' if n_hard == 0 else f'<span class="bad">有困难目标</span>'
        sam_sum = ""
        if sam_iou_cam or n_fallback_cam:
            med_iou = float(np.median(sam_iou_cam)) if sam_iou_cam else float("nan")
            iou_s = f"{med_iou:.3f}" if sam_iou_cam else "—"
            sam_sum = (
                f" | <b>SAM</b>：IoU 中位数 {html.escape(iou_s)}，"
                f"fallback {n_fallback_cam}/{len(objs)}"
            )

        img_block = (
            f'<img class="shot" src="{html.escape(assets_dir.name + "/" + img_rel)}?v={cache_bust}" alt="frame {fi}"/>'
            if img_rel
            else '<p class="muted">（无图像：路径缺失或 timesync 无此相机文件）</p>'
        )

        sections.append(
            f"""
<section class="frame-block" id="frame_{fi:06d}">
  <h2>帧 {fi:06d}</h2>
  {img_block}
  <div class="meta">
    <h3>帧信息</h3>
    <ul>
      <li><b>状态</b>：{status}</li>
      <li><b>摘要</b>：全帧目标: {len(objs_all)} | 本相机({html.escape(cam)}): {len(objs)} | 困难目标: {n_hard} | 最大难度: {max_diff:.3f}{sam_sum}</li>
      <li><b>Timestamp (ns)</b>：{html.escape(str(ts))}</li>
      <li><b>展示相机</b>：{html.escape(cam)} | <b>标定画幅</b>：{w_calib_tbl}×{h_calib_tbl}（2D 框已按磁盘图→标定尺寸缩放后与 3D 投影对齐）</li>
      <li><b>源 JSON</b>：{html.escape(demo_json.name)}（绝对路径见页眉）</li>
    </ul>
  </div>
  <table class="det">
    <thead>
      <tr>
        <th>#</th><th>类别</th><th>置信度</th><th>难度</th><th>等级</th>
        <th>视锥点数</th><th>仅框点数</th><th>SAM IoU</th><th>用 mask</th><th>回退</th>
        <th>密度</th><th>体积</th><th>检测相机</th>
      </tr>
    </thead>
    <tbody>
    {''.join(rows_html) if rows_html else '<tr><td colspan="13" class="muted">本相机无目标</td></tr>'}
    </tbody>
  </table>
</section>
"""
        )

    css = """
:root { --bg:#1a1d23; --panel:#252a33; --text:#e8eaed; --muted:#9aa0a6; --accent:#5c9ded; --hard:#5c2a2a; }
body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; line-height: 1.45; }
h1 { margin-top: 0; }
.toc { background: var(--panel); padding: 16px 20px; border-radius: 8px; margin-bottom: 28px; max-height: 220px; overflow-y: auto; }
.toc a { color: var(--accent); margin-right: 12px; display: inline-block; line-height: 1.8; }
.frame-block { margin-bottom: 48px; padding-bottom: 32px; border-bottom: 1px solid #333; }
.shot { max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #444; }
.meta { background: var(--panel); padding: 14px 18px; border-radius: 8px; margin: 14px 0; }
.meta ul { margin: 0; padding-left: 18px; }
.ok { color: #7bd88f; }
.bad { color: #ff7b7b; }
.muted { color: var(--muted); }
table.det { width: 100%; border-collapse: collapse; font-size: 14px; }
table.det th, table.det td { border: 1px solid #3d4450; padding: 8px 10px; text-align: left; }
table.det th { background: #2f3542; }
tr.row-hard { background: var(--hard); }
"""

    gen_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sam_banner = _sam_banner_html(meta, dbg_doc)
    dino_mode_s = html.escape(str(meta.get("dino_mode", "—")))
    prompt_s = html.escape(str(meta.get("prompt", "—"))[:200])
    demo_abs = html.escape(str(demo_json.resolve()))
    dbg_abs = html.escape(str(debug_json.resolve()))

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>自动标注报告 — {html.escape(demo_json.name)}</title>
<style>{css}</style>
</head>
<body>
<h1>自动标注 HTML 报告</h1>
<p class="muted">数据根目录：{html.escape(str(data_root.resolve()))}</p>
<p><b>标注 JSON</b>：<code>{demo_abs}</code><br/>
<b>内容指纹(前16)</b>：{html.escape(demo_sha)}（大小+mtime+首4MB） | <b>mtime</b>：{html.escape(demo_mtime)}</p>
<p><b>Debug JSON</b>：<code>{dbg_abs}</code><br/>
<b>内容指纹(前16)</b>：{html.escape(dbg_sha)} | <b>mtime</b>：{html.escape(dbg_mtime)}</p>
<p><b>报告生成时间</b>：{html.escape(gen_ts)} | <b>帧条数（去重 frame_idx）</b>：{len(sorted_fis)} | <b>重复 frame_idx 覆盖次数</b>：{n_dup} | <b>已删旧 asset 图</b>：{n_removed} 张</p>
<p><b>annotations 目标总数</b>：{n_objs_final} | <b>dino_mode</b>：{dino_mode_s}</p>
<p class="muted"><b>metadata.prompt</b>：{prompt_s}</p>
{sam_banner}
<p class="muted"><b>图层说明</b>：<span style="color:#ff8000">橙色线</span> 为 LiDAR 系 3D OBB 用标定 <code>K</code> 投影到 <b>标定 ImageSize</b> 像素平面；<span style="color:#ffc800">黄色细框</span> 为 JSON 中 DINO 的 <code>bbox_2d.xyxy</code>（原在磁盘图像分辨率下），报告已按宽高比 <b>缩放到与标定一致</b>，否则与橙线不在同一坐标系会看起来像「3D 没变」或错位。<b>表格</b>中 <b>视锥点数</b> 为与 lifting 一致的 LiDAR 点数；<b>仅框点数</b> / <b>SAM IoU</b> / <b>用 mask</b> / <b>回退</b> 来自同帧同相机 <code>raw_dino_debug</code> 中与该 2D 框 IoU 最大的检测行（需曾开启 <code>--sam-frustum</code> 等才有值）。</p>
<p class="muted">图片 URL 带 <code>?v=</code> 时间戳；asset 目录每次生成前会删除旧 <code>frame_*.jpg</code>，避免混入历史运行结果。</p>
<nav class="toc"><b>目录（{len(toc_items)} 帧）</b><br/>{' '.join(toc_items)}</nav>
{''.join(sections)}
</body>
</html>
"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(doc, encoding="utf-8")
    return output_html


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate HTML report from demo + debug JSON")
    ap.add_argument("--data-root", type=str, default="", help="数据集根目录（含 calibration.json、timesync、图像）")
    ap.add_argument("--demo-json", type=str, default="", help="annotations_demo.json（默认 post-NMS 最终标注）")
    ap.add_argument("--debug-json", type=str, default="", help="raw_dino_debug.json 路径")
    ap.add_argument("--output-html", type=str, default="", help="输出 report.html 路径")
    ap.add_argument("--output-root", type=str, default="", help="默认与 pipeline 一致，用于推断 demo/debug 与 report 路径")
    ap.add_argument(
        "--display-camera",
        type=str,
        default="front_left_camera",
        help="优先用于展示的相机（该帧无此相机检测时自动换用本帧出现的相机）",
    )
    ap.add_argument("--difficulty-thr", type=float, default=0.55)
    ap.add_argument("--max-table-rows", type=int, default=40)
    ap.add_argument(
        "--use-raw-annotations",
        action="store_true",
        help="未指定 --demo-json 时改用 annotations_demo_raw.json（NMS 前，3D 框更多）",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    from ..settings.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, AppConfig

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    out_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
    if args.demo_json:
        demo_json = Path(args.demo_json)
    elif args.use_raw_annotations:
        demo_json = out_root / "annotations_demo_raw.json"
    else:
        demo_json = out_root / "annotations_demo.json"
    debug_json = Path(args.debug_json) if args.debug_json else out_root / "raw_dino_debug.json"
    output_html = Path(args.output_html) if args.output_html else out_root / "report.html"

    cameras = AppConfig().cameras
    build_html_report(
        data_root=data_root,
        demo_json=demo_json,
        debug_json=debug_json,
        output_html=output_html,
        cameras=cameras,
        display_camera=args.display_camera,
        difficulty_thr=args.difficulty_thr,
        max_table_rows=args.max_table_rows,
    )
    print(f"已写入: {output_html.resolve()}")
    print(f"标注源: {demo_json.resolve()} (确认 SHA/mtime 与页眉一致)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
