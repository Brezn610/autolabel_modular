#!/usr/bin/env python3
"""从 stage_stats.json + raw_dino_debug.json 生成单页 HTML 运行汇总（无需 Open3D / 大图）。"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .console import _flatten_detections, _load_json, _safe_ratio


def _fmt_ratio(a: int, b: int) -> str:
    return f"{_safe_ratio(a, b):.2%}"


def _file_brief(path: Path) -> str:
    if not path.is_file():
        return "（不存在）"
    st = path.stat()
    return f"mtime={datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | size={st.st_size}"


def _score_line(name: str, scores: List[float]) -> str:
    if not scores:
        return f"<tr><td>{html.escape(name)}</td><td colspan=\"6\">无数据</td></tr>"
    arr = np.asarray(scores, dtype=np.float64)
    return (
        f"<tr><td>{html.escape(name)}</td>"
        f"<td>{arr.size}</td>"
        f"<td>{np.min(arr):.4f}</td>"
        f"<td>{np.median(arr):.4f}</td>"
        f"<td>{np.mean(arr):.4f}</td>"
        f"<td>{np.percentile(arr, 90):.4f}</td>"
        f"<td>{np.max(arr):.4f}</td></tr>"
    )


def _camera_rows(flat: List[Dict[str, Any]]) -> str:
    by_cam: Dict[str, List[Dict[str, Any]]] = {}
    for x in flat:
        by_cam.setdefault(str(x.get("camera", "")), []).append(x)
    rows: List[str] = []
    for cam in sorted(by_cam.keys()):
        items = by_cam[cam]
        n = len(items)
        n_fr = sum(int(i.get("frustum_points", 0)) > 0 for i in items)
        n_obb = sum(bool(i.get("obb_success")) for i in items)
        scores = [float(i.get("score", 0.0)) for i in items]
        ious = [
            float(i["sam_best_iou"])
            for i in items
            if isinstance(i.get("sam_best_iou"), (int, float))
        ]
        n_fb = sum(1 for i in items if i.get("sam_frustum_fallback"))
        n_used = sum(1 for i in items if i.get("sam_frustum_used_mask"))
        smed = f"{np.median(scores):.4f}" if scores else "—"
        imed = f"{np.median(ious):.4f}" if ious else "—"
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(cam)}</code></td>"
            f"<td>{n}</td>"
            f"<td>{_fmt_ratio(n_fr, n)}</td>"
            f"<td>{_fmt_ratio(n_obb, n)}</td>"
            f"<td>{smed}</td>"
            f"<td>{imed}</td>"
            f"<td>{n_used}</td>"
            f"<td>{n_fb}</td>"
            "</tr>"
        )
    return "\n".join(rows) if rows else "<tr><td colspan=\"8\">无检测展开数据</td></tr>"


def _stage_table(stage: Dict[str, Any]) -> str:
    sc = stage.get("stage_counts") or {}
    keys = ["raw_dets", "frustum_nonempty", "obb_success"]
    body = []
    for k in keys:
        body.append(f"<tr><td><code>{html.escape(k)}</code></td><td>{sc.get(k, '—')}</td></tr>")
    rd, fr, ob = int(sc.get("raw_dets", 0)), int(sc.get("frustum_nonempty", 0)), int(sc.get("obb_success", 0))
    body.append(
        f"<tr><td><code>frustum_nonempty / raw_dets</code></td><td>{_fmt_ratio(fr, rd)}</td></tr>"
    )
    body.append(f"<tr><td><code>obb_success / raw_dets</code></td><td>{_fmt_ratio(ob, rd)}</td></tr>")
    body.append(f"<tr><td><code>obb_success / frustum_nonempty</code></td><td>{_fmt_ratio(ob, fr)}</td></tr>")
    return "\n".join(body)


def _sam_frustum_table(sfs: Dict[str, Any]) -> str:
    if not sfs.get("enabled"):
        return "<p class=\"muted\">本次运行未开启 <code>sam_enabled_in_frustum</code>（<code>--sam-frustum</code>）。</p>"
    rows = []
    for k, v in sfs.items():
        rows.append(f"<tr><td><code>{html.escape(str(k))}</code></td><td>{html.escape(json.dumps(v))}</td></tr>")
    return "\n".join(rows)


def build_run_summary_html(
    stage_stats_path: Path,
    raw_debug_path: Path,
    output_html: Path,
) -> Path:
    stage = _load_json(stage_stats_path)
    raw_doc = _load_json(raw_debug_path) if raw_debug_path.is_file() else {}
    flat = _flatten_detections(raw_doc)

    sc = stage.get("stage_counts") or {}
    sfs = stage.get("sam_frustum_stats") or {}
    rk = stage.get("raw_kept_score_summary") or {}
    paths = stage.get("paths") or {}

    sam2_meta = raw_doc.get("sam2_meta") or {}
    sfm = raw_doc.get("sam_frustum_meta") or {}

    dino_scores = [float(x.get("score", 0.0)) for x in flat]
    sam_ious = [
        float(x["sam_best_iou"]) for x in flat if isinstance(x.get("sam_best_iou"), (int, float))
    ]
    fr_pts = [int(x.get("frustum_points", 0)) for x in flat]
    low_fr = sum(1 for p in fr_pts if p <= 3)

    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stage_brief = _file_brief(stage_stats_path)
    raw_brief = _file_brief(raw_debug_path)

    if rk:
        rk_rows = "".join(
            f"<tr><td><code>{html.escape(str(k))}</code></td><td>{html.escape(json.dumps(v))}</td></tr>"
            for k, v in rk.items()
        )
    else:
        rk_rows = '<tr><td colspan="2">无</td></tr>'

    low_fr_ratio = _fmt_ratio(low_fr, len(flat)) if flat else "—"

    css = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:960px;line-height:1.5;color:#1a1a1a;}
h1{font-size:1.45rem;border-bottom:1px solid #ddd;padding-bottom:8px;}
h2{font-size:1.1rem;margin-top:28px;}
table{border-collapse:collapse;width:100%;font-size:0.92rem;}
th,td{border:1px solid #ddd;padding:8px 10px;text-align:left;}
th{background:#f4f4f4;}
code{font-size:0.88em;background:#f0f0f0;padding:2px 5px;border-radius:4px;}
.muted{color:#666;font-size:0.9rem;}
ul.paths{list-style:none;padding:0;}
ul.paths li{margin:6px 0;word-break:break-all;}
"""

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>流水线运行汇总</title>
<style>{css}</style>
</head>
<body>
<h1>自动标注流水线 — 运行汇总</h1>
<p class="muted">生成时间（UTC）：{html.escape(gen_ts)}</p>

<h2>1. 输入文件</h2>
<table>
<tr><th>文件</th><th>说明</th></tr>
<tr><td><code>{html.escape(str(stage_stats_path.resolve()))}</code></td><td>{html.escape(stage_brief)}</td></tr>
<tr><td><code>{html.escape(str(raw_debug_path.resolve()))}</code></td><td>{html.escape(raw_brief)}</td></tr>
</table>

<h2>2. 阶段计数（stage_counts）</h2>
<table>
<tr><th>指标</th><th>值</th></tr>
{_stage_table(stage)}
</table>

<h2>3. SAM 视锥（sam_frustum_stats）</h2>
<table>
<tr><th>字段</th><th>值</th></tr>
{_sam_frustum_table(sfs)}
</table>

<h2>4. 分数与视锥（基于 raw_dino_debug 全量展开）</h2>
<p class="muted">展开检测条数：<b>{len(flat)}</b>；其中 <code>frustum_points≤3</code>：<b>{low_fr}</b>（{low_fr_ratio}）</p>
<table>
<tr><th>序列</th><th>N</th><th>min</th><th>median</th><th>mean</th><th>p90</th><th>max</th></tr>
{_score_line("DINO score", dino_scores)}
{_score_line("sam_best_iou", sam_ious)}
{_score_line("frustum_points", [float(p) for p in fr_pts])}
</table>

<h2>5. 保留标注分数摘要（raw_kept_score_summary，NMS 前保留集）</h2>
<table>
<tr><th>字段</th><th>值</th></tr>
{rk_rows}
</table>

<h2>6. 按相机分解</h2>
<table>
<tr>
<th>相机</th><th>det 数</th><th>frustum&gt;0</th><th>obb 成功</th>
<th>score 中位数</th><th>sam_iou 中位数</th><th>used_mask 条数</th><th>fallback 条数</th>
</tr>
{_camera_rows(flat)}
</table>

<h2>7. 元数据（raw_dino_debug 顶层）</h2>
<h3>sam2_meta</h3>
<pre class="muted" style="white-space:pre-wrap;background:#fafafa;padding:12px;border:1px solid #eee;border-radius:6px;">{html.escape(json.dumps(sam2_meta, ensure_ascii=False, indent=2))}</pre>
<h3>sam_frustum_meta</h3>
<pre class="muted" style="white-space:pre-wrap;background:#fafafa;padding:12px;border:1px solid #eee;border-radius:6px;">{html.escape(json.dumps(sfm, ensure_ascii=False, indent=2))}</pre>

<h2>8. 产物路径（paths）</h2>
<ul class="paths">
{"".join(f"<li><b>{html.escape(str(k))}</b>：<code>{html.escape(str(v))}</code></li>" for k, v in paths.items())}
</ul>
<p class="muted">完整逐帧可视化见 <code>report.html</code>（需 annotations + 大图）；DINO 单页见 <code>dino_report.html</code>；SAM 调试图见 <code>sam_debug/</code>。</p>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(doc, encoding="utf-8")
    return output_html


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="从 stage_stats + raw_dino_debug 生成 HTML 运行汇总")
    ap.add_argument("--output-root", type=str, default="", help="输出根目录（默认 DEMO_OUT / 内置 out）")
    ap.add_argument("--stage-stats", type=str, default="", help="stage_stats.json 路径")
    ap.add_argument("--raw-debug", type=str, default="", help="raw_dino_debug.json 路径")
    ap.add_argument("--output-html", type=str, default="", help="输出 HTML，默认 output-root/run_summary_report.html")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    from ..settings.config import DEFAULT_OUTPUT_ROOT

    out_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
    stage_path = Path(args.stage_stats) if args.stage_stats else out_root / "stage_stats.json"
    raw_path = Path(args.raw_debug) if args.raw_debug else out_root / "raw_dino_debug.json"
    if not raw_path.is_file() and stage_path.is_file():
        try:
            alt = Path(json.loads(stage_path.read_text(encoding="utf-8")).get("paths", {}).get("raw_dino_debug", ""))
            if alt.is_file():
                raw_path = alt
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    out_html = Path(args.output_html) if args.output_html else out_root / "run_summary_report.html"

    build_run_summary_html(stage_path, raw_path, out_html)
    print(f"已写入: {out_html.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
