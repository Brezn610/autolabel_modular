#!/usr/bin/env python3
"""仅根据 raw_dino_debug.json 生成单页 HTML：每帧配图 + DINO 框 + 检测表（不依赖 annotations）。"""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from autolabel_modular.calib.calibration import get_image_size_hw, load_and_validate_calibration
from autolabel_modular.ingest.timesync import load_timesync_table
from autolabel_modular.settings.config import AppConfig


def _dino_difficulty(frustum_points: int, score: float) -> float:
    p = max(0, int(frustum_points))
    pn = 1.0 - min(1.0, p / 200.0)
    sn = 1.0 - float(np.clip(score, 0.0, 1.0))
    return float(np.clip(0.55 * pn + 0.45 * sn, 0.0, 1.0))


def _density(points: int, box_xyxy: List[float]) -> float:
    x1, y1, x2, y2 = box_xyxy
    area = max(1.0, (x2 - x1) * (y2 - y1))
    return float(points) / area * 1e4


def _pick_display_camera(
    fr_dbg: Dict[str, Any], preferred: str, cameras: List[str]
) -> str:
    cams = fr_dbg.get("cameras") or {}
    pref_dets = (cams.get(preferred) or {}).get("detections") or []
    if pref_dets:
        return preferred
    best_cam = preferred
    best_n = -1
    for cam in cameras:
        n = len((cams.get(cam) or {}).get("detections") or [])
        if n > best_n:
            best_n = n
            best_cam = cam
    return best_cam


def _draw_dino_overlay(
    img_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    colors_bgr: Dict[str, Tuple[int, int, int]],
    default_bgr: Tuple[int, int, int] = (0, 200, 255),
) -> np.ndarray:
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]
    for det in detections:
        box = det.get("box_xyxy")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(t) for t in box]
        x1 = max(0.0, min(x1, w - 1.0))
        x2 = max(0.0, min(x2, w - 1.0))
        y1 = max(0.0, min(y1, h - 1.0))
        y2 = max(0.0, min(y2, h - 1.0))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            continue
        lab = str(det.get("label", "")).strip().lower()
        col = colors_bgr.get(lab, default_bgr)
        ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), col, 2, lineType=cv2.LINE_AA)
        sc = float(det.get("score", 0.0))
        txt = f"{lab} {sc:.2f}"
        cv2.putText(
            canvas,
            txt[:40],
            (ix1, max(14, iy1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_dino_html_report(
    data_root: Path,
    debug_json: Path,
    output_html: Path,
    cameras: List[str],
    display_camera: str,
    max_frames: int,
    difficulty_thr: float,
    max_table_rows: int,
) -> Path:
    with debug_json.open("r", encoding="utf-8") as f:
        dbg_doc = json.load(f)

    raw_frames: List[Dict[str, Any]] = list(dbg_doc.get("frames") or [])
    raw_frames.sort(key=lambda fr: int(fr.get("frame_idx", 0)))
    frames_slice = raw_frames[: max(0, int(max_frames))]

    calib_bundle = load_and_validate_calibration(data_root / "calibration.json", cameras)
    calib = calib_bundle.raw

    need_ts = 1
    if frames_slice:
        need_ts = int(frames_slice[-1].get("frame_idx", 0)) + 1
    ts_frames = load_timesync_table(data_root, max_frames=max(need_ts, 1))

    cfg_colors = AppConfig().class_colors_bgr
    colors_bgr = {k.lower(): tuple(v) for k, v in cfg_colors.items()}

    assets_dir = output_html.parent / (output_html.stem + "_assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 全局统计（当前切片内）
    label_ctr: Counter[str] = Counter()
    n_dets = 0
    n_obb_ok = 0
    for fr in frames_slice:
        for cam in cameras:
            for d in ((fr.get("cameras") or {}).get(cam) or {}).get("detections") or []:
                n_dets += 1
                label_ctr[str(d.get("label", "?")).lower()] += 1
                if d.get("obb_success"):
                    n_obb_ok += 1

    toc_items: List[str] = []
    sections: List[str] = []

    for fr in frames_slice:
        fi = int(fr.get("frame_idx", 0))
        cam = _pick_display_camera(fr, display_camera, cameras)
        toc_items.append(f'<a href="#frame_{fi:06d}">帧 {fi:06d}</a>')

        cams_dbg = fr.get("cameras") or {}
        cam_block = cams_dbg.get(cam) or {}
        dets: List[Dict[str, Any]] = list(cam_block.get("detections") or [])

        img_rel = ""
        if fi < len(ts_frames) and cam in ts_frames[fi].files:
            img_path = data_root / cam / ts_frames[fi].files[cam]
            if img_path.is_file():
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is not None:
                    intr = calib[cam]["intrinsics"]
                    h_calib, w_calib = get_image_size_hw(intr)
                    ih, iw = img.shape[:2]
                    if (ih, iw) != (h_calib, w_calib):
                        img = cv2.resize(img, (w_calib, h_calib), interpolation=cv2.INTER_LINEAR)
                    vis = _draw_dino_overlay(img, dets, colors_bgr)
                    out_img = assets_dir / f"frame_{fi:06d}__{cam}.jpg"
                    cv2.imwrite(str(out_img), vis)
                    img_rel = out_img.name

        rows_html: List[str] = []
        difficulties: List[float] = []
        table_dets = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)[:max_table_rows]

        for idx, d in enumerate(table_dets, start=1):
            lab = str(d.get("label", ""))
            score = float(d.get("score", 0.0))
            pts = int(d.get("frustum_points", 0))
            obb_ok = bool(d.get("obb_success"))
            bx = d.get("box_xyxy") or [0, 0, 0, 0]
            if len(bx) != 4:
                bx = [0, 0, 0, 0]
            dens = _density(pts, [float(x) for x in bx])
            diff = _dino_difficulty(pts, score)
            difficulties.append(diff)

            if diff >= difficulty_thr:
                level = "困难（低点数/低分）"
                row_cls = "row-hard"
            else:
                level = "正常"
                row_cls = ""

            rows_html.append(
                "<tr class='{cls}'>"
                "<td>{idx}</td><td>{lab}</td><td>{score:.3f}</td><td>{diff:.3f}</td>"
                "<td>{level}</td><td>{pts}</td><td>{dens:.4f}</td><td>{obb}</td>"
                "</tr>".format(
                    cls=row_cls,
                    idx=idx,
                    lab=html.escape(lab),
                    score=score,
                    diff=diff,
                    level=html.escape(level),
                    pts=pts,
                    dens=dens,
                    obb=html.escape("是" if obb_ok else "否"),
                )
            )

        n_hard = sum(1 for x in difficulties if x >= difficulty_thr)
        max_diff = max(difficulties) if difficulties else 0.0
        status = f'<span class="ok">正常</span>' if n_hard == 0 else f'<span class="bad">有困难检测</span>'

        n_cam_all = sum(
            len((cams_dbg.get(c) or {}).get("detections") or []) for c in cameras
        )

        img_block = (
            f'<img class="shot" src="{html.escape(assets_dir.name + "/" + img_rel)}" alt="frame {fi}"/>'
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
      <li><b>摘要</b>：本帧全相机检测数 {n_cam_all} | 展示相机({html.escape(cam)})：{len(dets)} | 表内困难行: {n_hard} | 最大难度: {max_diff:.3f}</li>
      <li><b>展示相机</b>：{html.escape(cam)}</li>
      <li><b>源 JSON</b>：{html.escape(debug_json.name)}</li>
    </ul>
  </div>
  <table class="det">
    <thead>
      <tr>
        <th>#</th><th>标签</th><th>置信度</th><th>难度</th><th>等级</th>
        <th>视锥点数</th><th>密度</th><th>OBB 成功</th>
      </tr>
    </thead>
    <tbody>
    {''.join(rows_html) if rows_html else '<tr><td colspan="8" class="muted">本相机无检测</td></tr>'}
    </tbody>
  </table>
</section>
"""
        )

    label_summary = ", ".join(f"{k}: {v}" for k, v in label_ctr.most_common()) or "无"
    obb_rate = (n_obb_ok / n_dets * 100.0) if n_dets else 0.0

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
.summary { background: var(--panel); padding: 16px 20px; border-radius: 8px; margin-bottom: 24px; }
"""

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Grounding DINO 报告 — {html.escape(debug_json.name)}</title>
<style>{css}</style>
</head>
<body>
<h1>Grounding DINO 2D 检测 HTML 报告</h1>
<p class="muted">数据根目录：{html.escape(str(data_root))} | Debug：{html.escape(str(debug_json))}</p>
<div class="summary">
  <b>切片统计</b>（至多 {len(frames_slice)} 帧）<br/>
  总检测数：{n_dets} | OBB 成功：{n_obb_ok}（{obb_rate:.1f}%）<br/>
  按标签：{html.escape(label_summary)}
</div>
<p class="muted">彩色框为 DINO 2D 框（颜色与 AppConfig 类别色一致；未知标签为默认色）。难度为启发式：视锥点数偏少、置信度偏低则升高，便于扫一眼找弱检。</p>
<nav class="toc"><b>目录（{len(toc_items)} 帧）</b><br/>{' '.join(toc_items)}</nav>
{''.join(sections)}
</body>
</html>
"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(doc, encoding="utf-8")
    return output_html


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="仅从 raw_dino_debug.json 生成 HTML 报告（无需 annotations）")
    ap.add_argument("--data-root", type=str, default="", help="数据集根（calibration.json、timesync、相机图）")
    ap.add_argument("--debug-json", type=str, default="", help="raw_dino_debug.json")
    ap.add_argument("--output-html", type=str, default="", help="输出 HTML，默认 <output-root>/dino_report.html")
    ap.add_argument("--output-root", type=str, default="", help="推断默认路径")
    ap.add_argument("--display-camera", type=str, default="front_left_camera", help="优先展示的相机")
    ap.add_argument("--max-frames", type=int, default=100, help="报告中包含的帧数上限（按 frame_idx 排序后截取）")
    ap.add_argument("--difficulty-thr", type=float, default=0.55)
    ap.add_argument("--max-table-rows", type=int, default=40)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    from autolabel_modular.settings.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    out_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
    debug_json = Path(args.debug_json) if args.debug_json else out_root / "raw_dino_debug.json"
    output_html = Path(args.output_html) if args.output_html else out_root / "dino_report.html"

    if not debug_json.is_file():
        raise FileNotFoundError(debug_json)

    cameras = AppConfig().cameras
    build_dino_html_report(
        data_root=data_root,
        debug_json=debug_json,
        output_html=output_html,
        cameras=cameras,
        display_camera=args.display_camera,
        max_frames=args.max_frames,
        difficulty_thr=args.difficulty_thr,
        max_table_rows=args.max_table_rows,
    )
    print(f"已写入: {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
