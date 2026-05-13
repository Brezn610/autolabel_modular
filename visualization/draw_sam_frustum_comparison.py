#!/usr/bin/env python3
"""
SAM2 接入视锥后的对比可视化：原图 + DINO 框 + SAM mask + 视锥 LiDAR 点。

推荐在仓库根目录执行（使 ``calib`` / ``ingest`` 等包在路径上）::

    cd /path/to/autolabel_modular
    python -m visualization.draw_sam_frustum_comparison --output-root ./out --max-frames 50 --cameras all

或使用包路径::

    cd /path/to/parent_of_repo
    PYTHONPATH=. python -m autolabel_modular.visualization.draw_sam_frustum_comparison ...
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# 支持: (1) cd autolabel_modular && python -m visualization.*  (2) PYTHONPATH=父目录 python -m autolabel_modular.visualization.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_PARENT = _REPO_ROOT.parent
for p in (_PKG_PARENT, _REPO_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from autolabel_modular.calib.calibration import (
    build_distortion_coeffs,
    get_T_lidar_to_cam,
    get_image_size_hw,
    load_and_validate_calibration,
    parse_intrinsic_to_K,
)
from autolabel_modular.detect.sam2_masks import cv2_resize_bool, zlib_b64_to_mask_bool
from autolabel_modular.detect.sam_refinement import apply_sam_mask_to_frustum
from autolabel_modular.geom.projection import project_lidar_to_image
from autolabel_modular.ingest.lidar_io import load_lidar_xyz
from autolabel_modular.ingest.timesync import load_timesync_table
from autolabel_modular.settings.config import AppConfig, DEFAULT_DATA_ROOT

COLOR_DINO_BOX = (0, 140, 255)  # 橙 BGR
COLOR_MASK = (0, 220, 0)  # 绿
COLOR_PT_OK = (255, 255, 255)
COLOR_PT_FB = (60, 60, 255)  # 红 BGR
MAX_PTS_TOTAL = 8000


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mask_from_det(det: Dict[str, Any]) -> Optional[np.ndarray]:
    b64 = det.get("sam_mask_zlib_b64")
    shp = det.get("sam_mask_shape_hw")
    if b64 and shp and len(shp) == 2:
        try:
            return zlib_b64_to_mask_bool(str(b64), [int(shp[0]), int(shp[1])])
        except (ValueError, OSError, EOFError, TypeError, zlib.error):
            return None
    return None


def _overlay_mask_bgr(canvas: np.ndarray, mask: np.ndarray, bgr: Tuple[int, int, int], alpha: float) -> None:
    m = mask.astype(bool)
    if not np.any(m):
        return
    fl = canvas.astype(np.float32)
    b, g, r = bgr[0], bgr[1], bgr[2]
    cols = (b, g, r)
    for c in range(3):
        ch = fl[:, :, c]
        col = cols[c]
        ch[m] = ch[m] * (1.0 - alpha) + float(col) * alpha
    np.clip(fl, 0, 255, out=fl)
    canvas[:] = fl.astype(np.uint8)


def _put_text_outline(
    img: np.ndarray,
    lines: List[str],
    org: Tuple[int, int],
    *,
    font_scale: float = 0.65,
    thickness: int = 1,
    fg: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org
    dy = int(22 * font_scale + 8)
    for line in lines:
        pos = (x, y)
        for ox, oy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.putText(
                img,
                line,
                (pos[0] + ox, pos[1] + oy),
                font,
                font_scale,
                bg,
                thickness + 1,
                cv2.LINE_AA,
            )
        cv2.putText(img, line, pos, font, font_scale, fg, thickness, cv2.LINE_AA)
        y += dy


def _put_fallback_banner(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    text = "FALLBACK"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = min(w, h) / 500.0 * 1.2
    (tw, th), _ = cv2.getTextSize(text, font, scale, 3)
    x = w - tw - 24
    y = th + 24
    for ox, oy in ((-2, -2), (-2, 2), (2, -2), (2, 2), (-2, 0), (2, 0), (0, -2), (0, 2)):
        cv2.putText(img, text, (x + ox, y + oy), font, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (60, 60, 255), 3, cv2.LINE_AA)


def _parse_cameras(arg: str, fallback: List[str]) -> List[str]:
    s = (arg or "all").strip().lower()
    if s in ("", "all"):
        return list(fallback)
    return [c.strip() for c in arg.split(",") if c.strip()]


def _parse_int_set(arg: str) -> Set[int]:
    out: Set[int] = set()
    for p in (arg or "").split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except ValueError:
            pass
    return out


def _debug_frame(raw_doc: Dict[str, Any], fi: int) -> Dict[str, Any]:
    for fr in raw_doc.get("frames") or []:
        if int(fr.get("frame_idx", -1)) == fi:
            return fr
    return {}


def draw_comparison_for_frame_camera(
    bgr: np.ndarray,
    dets: List[Dict[str, Any]],
    uv: np.ndarray,
    z_cam: np.ndarray,
    calib_hw: Tuple[int, int],
    z_min: float,
    z_max: float,
    dilate_iters: int,
    min_points: int,
    cam: str,
    fi: int,
    scale: int,
) -> Tuple[np.ndarray, bool]:
    """
    返回 (BGR 图, 是否含任一 FALLBACK)。
    绘制顺序：全部 SAM mask → 全部 DINO 框 → 全部视锥点。
    """
    h0, w0 = bgr.shape[:2]
    ch, cw = calib_hw[0], calib_hw[1]
    if ch <= 0 or cw <= 0:
        canvas = bgr.copy()
    elif (h0, w0) != (ch, cw):
        canvas = cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_LINEAR)
    else:
        canvas = bgr.copy()

    canvas_f = canvas.astype(np.float32)
    any_fallback = bool(any(d.get("sam_frustum_fallback") for d in dets))
    n_det = max(1, len(dets))
    per_cap = max(1, MAX_PTS_TOTAL // n_det)

    def _norm_box(box: List[float]) -> List[float]:
        bx = [float(t) for t in box]
        if (h0, w0) != (ch, cw):
            sx = cw / float(w0)
            sy = ch / float(h0)
            bx = [bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy]
        return bx

    # 1) 所有 SAM mask（半透明绿）
    for det in dets:
        box = det.get("box_xyxy")
        if not box or len(box) != 4:
            continue
        m_arr = _mask_from_det(det)
        if m_arr is not None and m_arr.size > 0:
            mm = m_arr
            if mm.shape[0] != ch or mm.shape[1] != cw:
                mm = cv2_resize_bool(mm, (cw, ch))
            _overlay_mask_bgr(canvas_f, mm, COLOR_MASK, 0.4)

    # 2) 所有 DINO 框（橙实线）
    for det in dets:
        box = det.get("box_xyxy")
        if not box or len(box) != 4:
            continue
        bx = _norm_box(list(box))
        x1, y1, x2, y2 = [int(round(t)) for t in bx]
        cv2.rectangle(canvas_f, (x1, y1), (x2, y2), COLOR_DINO_BOX, 2, cv2.LINE_AA)

    # 3) 视锥 LiDAR 点（白 / 红）
    for det in dets:
        box = det.get("box_xyxy")
        if not box or len(box) != 4:
            continue
        bx = _norm_box(list(box))
        m_arr = _mask_from_det(det)
        mask_bool, _meta = apply_sam_mask_to_frustum(
            uv,
            z_cam,
            bx,
            z_min,
            z_max,
            m_arr,
            (ch, cw),
            dilate_iters=dilate_iters,
            min_points=min_points,
        )
        pts_idx = mask_bool & np.isfinite(uv).all(axis=1)
        uv_sel = uv[pts_idx]
        if uv_sel.shape[0] > per_cap:
            rng = np.random.default_rng(42 + fi + hash(cam) % 997)
            pick = rng.choice(uv_sel.shape[0], size=per_cap, replace=False)
            uv_sel = uv_sel[pick]
        fb = bool(det.get("sam_frustum_fallback"))
        col = COLOR_PT_FB if fb else COLOR_PT_OK
        for i in range(uv_sel.shape[0]):
            u, v = int(round(float(uv_sel[i, 0]))), int(round(float(uv_sel[i, 1])))
            if 0 <= u < cw and 0 <= v < ch:
                cv2.circle(canvas_f, (u, v), 2, col, -1, cv2.LINE_AA)

    out = np.clip(canvas_f, 0, 255).astype(np.uint8)

    lines_head = [f"Camera: {cam} | Frame: {fi}"]
    if dets:
        scores = [float(d.get("score", 0.0)) for d in dets]
        sm = float(np.median(scores)) if scores else 0.0
        d0 = dets[0]
        n_box0 = int(d0.get("frustum_points_box_only", d0.get("frustum_points", 0)))
        n_fin0 = int(d0.get("frustum_points", 0))
        lines_head.append(f"DINO score (median / 1st): {sm:.3f} / {float(d0.get('score', 0)):.3f}")
        lines_head.append(f"Points (1st det): {n_box0} -> {n_fin0} (box -> final)")
        n_fb = sum(1 for d in dets if d.get("sam_frustum_fallback"))
        if n_fb == 0:
            st = "SAM used"
        elif n_fb >= len(dets):
            st = "FALLBACK (all)"
        else:
            st = f"Mixed ({n_fb}/{len(dets)} FALLBACK)"
        lines_head.append(f"Status: {st} | dets={len(dets)}")
    else:
        lines_head.append("No detections")

    _put_text_outline(out, lines_head, (12, 28), font_scale=0.7, thickness=1)

    if any_fallback:
        _put_fallback_banner(out)

    if scale > 1:
        out = cv2.resize(out, (out.shape[1] * scale, out.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

    return out, any_fallback


def build_index_html(entries: List[Dict[str, str]], out_path: Path, highlights: List[str]) -> None:
    """entries: rel_path full image, rel_path thumb, title"""
    cards = []
    for e in entries:
        cards.append(
            f'<figure class="card"><a href="{html.escape(e["full"])}">'
            f'<img src="{html.escape(e["thumb"])}" alt="{html.escape(e["title"])}"/></a>'
            f'<figcaption>{html.escape(e["title"])}</figcaption></figure>'
        )
    hl = ""
    if highlights:
        hl_items = "".join(
            f'<li><a href="{html.escape(h)}">{html.escape(Path(h).name)}</a></li>' for h in highlights
        )
        hl = f"<h2>重点图（问题帧 / 后视）</h2><ul>{hl_items}</ul>"

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SAM–Frustum 可视化索引</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #111; color: #eee; }}
h1 {{ font-size: 1.35rem; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
.card {{ width: 280px; background: #1e1e1e; border-radius: 8px; padding: 8px; }}
.card img {{ width: 100%; height: auto; border-radius: 4px; cursor: zoom-in; }}
figcaption {{ font-size: 0.82rem; margin-top: 6px; color: #bbb; }}
a {{ color: #7ec8ff; }}
ul {{ line-height: 1.8; }}
</style>
</head>
<body>
<h1>SAM2 视锥对比图索引</h1>
<p>点击缩略图查看大图。橙色框 = DINO；绿色半透明 = SAM mask（若 JSON 含 zlib mask）；白/红点 = 视锥 LiDAR。</p>
{hl}
<h2>全部生成图</h2>
<div class="grid">
{"".join(cards)}
</div>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")


def run(
    data_root: Path,
    output_root: Path,
    raw_debug_path: Path,
    max_frames: int,
    cameras: List[str],
    highlight_frames: Set[int],
    highlight_cameras: Set[str],
    scale: int,
) -> Path:
    raw_doc = _load_json(raw_debug_path)
    meta_top = raw_doc.get("sam_frustum_meta") or {}
    s2 = raw_doc.get("sam2_meta") or {}
    _def = AppConfig()
    min_points = int(meta_top.get("fallback_min_points", _def.sam_frustum_fallback_min_points))
    dilate_iters = int(meta_top.get("mask_dilate_iters", _def.sam_frustum_mask_dilate_iters))
    z_min = float(_def.frustum_z_min)
    z_max = float(_def.frustum_z_max)

    cfg = AppConfig(data_root=data_root)
    calib_bundle = load_and_validate_calibration(data_root / "calibration.json", cfg.cameras)
    calib = calib_bundle.raw

    ts = load_timesync_table(data_root, max_frames=max(1, max_frames))
    n_frames = min(max_frames, len(ts))

    vis_root = output_root / "sam_frustum_visual"
    img_dir = vis_root / "compare"
    thumb_dir = vis_root / "thumbs"
    hl_dir = vis_root / "highlights"
    for d in (img_dir, thumb_dir, hl_dir):
        d.mkdir(parents=True, exist_ok=True)

    cams_use = [c for c in cameras if c in calib]
    if not cams_use:
        raise ValueError("无有效相机（检查标定与 --cameras）")

    index_entries: List[Dict[str, str]] = []
    highlight_rel: List[str] = []

    for fi in range(n_frames):
        fr_ts = ts[fi]
        lidar_name = fr_ts.files.get("middle_lidar")
        if not lidar_name:
            continue
        xyz = load_lidar_xyz(data_root, lidar_name)

        fr_dbg = _debug_frame(raw_doc, fi)
        cams_dbg = fr_dbg.get("cameras") or {}

        for cam in cams_use:
            if cam not in fr_ts.files:
                continue
            img_path = data_root / cam / fr_ts.files[cam]
            if not img_path.is_file():
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue

            intr = calib[cam]["intrinsics"]
            K = parse_intrinsic_to_K(intr)
            dist = build_distortion_coeffs(intr)
            T_l2c = get_T_lidar_to_cam(calib, cam)
            ch, cw = get_image_size_hw(intr)
            uv, zc = project_lidar_to_image(xyz.astype(np.float64), T_l2c, K, dist)

            ci = cams_dbg.get(cam) or {}
            dets: List[Dict[str, Any]] = list(ci.get("detections") or [])

            canvas, any_fb = draw_comparison_for_frame_camera(
                bgr,
                dets,
                uv,
                zc,
                (ch, cw),
                z_min,
                z_max,
                dilate_iters,
                min_points,
                cam,
                fi,
                scale,
            )

            fname = f"frame_{fi:06d}__{cam}.jpg"
            full_p = img_dir / fname
            cv2.imwrite(str(full_p), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            th = cv2.resize(canvas, (320, int(320 * canvas.shape[0] / max(1, canvas.shape[1]))))
            thumb_p = thumb_dir / fname
            cv2.imwrite(str(thumb_p), th, [int(cv2.IMWRITE_JPEG_QUALITY), 82])

            rel_full = "compare/" + fname
            rel_thumb = "thumbs/" + fname
            title = f"{cam} | frame {fi} | dets={len(dets)}"
            index_entries.append({"full": rel_full, "thumb": rel_thumb, "title": title})

            if fi in highlight_frames or cam in highlight_cameras:
                hl_name = f"frame_{fi:06d}__{cam}.jpg"
                hl_path = hl_dir / hl_name
                cv2.imwrite(str(hl_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                highlight_rel.append("highlights/" + hl_name)

    meta_side = {
        "sam_debug_dir": s2.get("sam_debug_dir", ""),
        "raw_dino_debug": str(raw_debug_path.resolve()),
        "data_root": str(data_root.resolve()),
        "note": "若未开启 sam_mask_zlib_b64，则仅绘制框+LiDAR 点，无绿色 mask 填充。",
    }
    (vis_root / "run_meta.json").write_text(
        json.dumps(meta_side, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    build_index_html(index_entries, vis_root / "index.html", highlight_rel)
    return vis_root


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SAM–Frustum 对比可视化 + index.html")
    ap.add_argument("--output-root", type=str, default="./out", help="含 raw_dino_debug.json 的输出根目录")
    ap.add_argument("--data-root", type=str, default="", help="数据集根（默认从 annotations metadata 推断）")
    ap.add_argument("--raw-debug", type=str, default="", help="raw_dino_debug.json 路径")
    ap.add_argument("--annotations", type=str, default="", help="annotations_demo.json，用于推断 data_root")
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--cameras", type=str, default="all", help="逗号分隔相机名，或 all")
    ap.add_argument(
        "--highlight-frames",
        type=str,
        default="19,12,13,5,7",
        help="重点帧（逗号分隔），额外写入 highlights/",
    )
    ap.add_argument(
        "--highlight-cameras",
        type=str,
        default="back_left_camera,back_right_camera",
        help="重点相机，额外写入 highlights/",
    )
    ap.add_argument("--scale", type=int, default=1, help="输出放大倍数（1=标定分辨率）")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.output_root).resolve()
    raw_path = Path(args.raw_debug) if args.raw_debug else out_root / "raw_dino_debug.json"
    if not raw_path.is_file():
        print(f"找不到 {raw_path}", file=sys.stderr)
        return 1

    data_root: Optional[Path] = None
    if args.data_root:
        data_root = Path(args.data_root).resolve()
    ann_path = Path(args.annotations) if args.annotations else out_root / "annotations_demo.json"
    if data_root is None and ann_path.is_file():
        try:
            dj = _load_json(ann_path)
            dr = (dj.get("metadata") or {}).get("dataset_root")
            if dr:
                data_root = Path(str(dr)).resolve()
        except (OSError, json.JSONDecodeError):
            pass
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT.resolve()

    if not (data_root / "calibration.json").is_file():
        print(f"数据根缺少 calibration.json: {data_root}", file=sys.stderr)
        return 1

    cfg = AppConfig()
    cams = _parse_cameras(args.cameras, cfg.cameras)
    hf = _parse_int_set(args.highlight_frames)
    hc = {s.strip() for s in (args.highlight_cameras or "").split(",") if s.strip()}

    vis = run(
        data_root=data_root,
        output_root=out_root,
        raw_debug_path=raw_path,
        max_frames=int(args.max_frames),
        cameras=cams,
        highlight_frames=hf,
        highlight_cameras=hc,
        scale=max(1, int(args.scale)),
    )
    print(f"已写入: {vis.resolve()}")
    print(f"  对比图: {vis / 'compare'}")
    print(f"  缩略图: {vis / 'thumbs'}")
    print(f"  重点图: {vis / 'highlights'}")
    print(f"  索引页: {(vis / 'index.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
