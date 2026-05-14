from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


BG = (28, 31, 36)
PANEL = (42, 47, 55)
TEXT = (235, 238, 242)
MUTED = (175, 183, 194)
ACCENT = {
    "accepted": (90, 210, 120),
    "recovered": (0, 165, 255),
    "uncertain": (80, 180, 255),
    "rejected": (80, 80, 230),
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], scale: float = 0.65, color: Tuple[int, int, int] = TEXT, thickness: int = 1) -> None:
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _card(img: np.ndarray, xyxy: Tuple[int, int, int, int], title: str) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), PANEL, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (65, 72, 82), 1, cv2.LINE_AA)
    _draw_text(img, title, (x1 + 18, y1 + 34), 0.72, TEXT, 2)


def _bar(img: np.ndarray, x: int, y: int, w: int, h: int, value: float, color: Tuple[int, int, int], label: str) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), (58, 64, 72), -1, cv2.LINE_AA)
    fill = int(max(0.0, min(1.0, value)) * w)
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1, cv2.LINE_AA)
    _draw_text(img, label, (x, y - 8), 0.54, MUTED, 1)


def render_fusion_summary(fusion_tracks_path: Path, output_dir: Path) -> Path:
    doc = _read_json(fusion_tracks_path)
    summary = doc["summary"]
    img = np.full((860, 1280, 3), BG, dtype=np.uint8)
    _draw_text(img, "Fusion Summary", (36, 56), 1.25, TEXT, 2)
    _draw_text(img, "Detector-driven pseudo-label status counts", (38, 88), 0.62, MUTED, 1)

    _card(img, (36, 120, 1244, 310), "Track Status")
    counts = [
        ("accepted", int(summary["accepted"])),
        ("recovered", int(summary["recovered"])),
        ("uncertain", int(summary["uncertain"])),
        ("rejected", int(summary["rejected"])),
    ]
    total = int(summary["tracks_total"])
    x = 70
    for status, count in counts:
        cv2.rectangle(img, (x, 168), (x + 250, 268), ACCENT[status], -1, cv2.LINE_AA)
        _draw_text(img, status, (x + 18, 202), 0.72, (15, 18, 22), 2)
        _draw_text(img, str(count), (x + 18, 248), 1.12, (15, 18, 22), 2)
        _draw_text(img, f"{100.0 * count / max(1, total):.1f}% of tracks", (x + 18, 292), 0.48, MUTED, 1)
        x += 292
    _draw_text(img, f"Total tracks: {total}", (70, 342), 0.72, TEXT, 2)
    _draw_text(img, f"Frame-level pseudo label objects: {summary.get('frame_level_pseudo_label_objects', 0)}", (360, 342), 0.72, TEXT, 2)

    _card(img, (36, 380, 1244, 790), "Per-Class Status Counts")
    y = 450
    for cls, row in sorted(summary["per_class"].items()):
        cls_total = int(row["total"])
        _draw_text(img, f"{cls}: total {cls_total}", (70, y), 0.75, TEXT, 2)
        yy = y + 44
        for status in ("accepted", "recovered", "uncertain", "rejected"):
            count = int(row[status])
            _bar(
                img,
                260,
                yy - 19,
                720,
                22,
                count / max(1, cls_total),
                ACCENT[status],
                f"{status}: {count}",
            )
            _draw_text(img, f"{100.0 * count / max(1, cls_total):.1f}%", (1005, yy), 0.52, MUTED, 1)
            yy += 48
        y += 185

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "fusion_summary.png"
    cv2.imwrite(str(out), img)
    return out


def render_evaluation_summary(evaluation_path: Path, output_dir: Path) -> Path:
    doc = _read_json(evaluation_path)
    summary = doc["summary"]
    gain = doc["retrieval_gain"]
    img = np.full((920, 1280, 3), BG, dtype=np.uint8)
    _draw_text(img, "Evaluation Summary", (36, 56), 1.25, TEXT, 2)
    _draw_text(img, "Center-distance evaluation against normalized DrivIng annotations", (38, 88), 0.62, MUTED, 1)

    methods = [
        "accepted_only",
        "accepted_plus_recovered",
        "accepted_recovered_uncertain",
        "recovered_only",
    ]
    _card(img, (36, 120, 1244, 585), "Ablation Metrics")
    x0, y0 = 74, 190
    col_w = 292
    for idx, method in enumerate(methods):
        row = summary[method]
        x = x0 + idx * col_w
        cv2.rectangle(img, (x, y0), (x + 250, y0 + 330), (50, 56, 65), -1, cv2.LINE_AA)
        _draw_text(img, method.replace("_", " "), (x + 14, y0 + 32), 0.52, TEXT, 1)
        metrics = [
            ("precision", float(row["precision"])),
            ("recall", float(row["recall"])),
            ("F1", float(row["f1"])),
        ]
        yy = y0 + 82
        for name, value in metrics:
            _bar(img, x + 22, yy, 205, 24, value, (90, 190, 255), f"{name}: {value:.3f}")
            yy += 72
        _draw_text(img, f"TP {row['tp']}  FP {row['fp']}", (x + 22, y0 + 284), 0.56, MUTED, 1)
        _draw_text(img, f"FN {row['fn']}  pred {row['predictions']}", (x + 22, y0 + 312), 0.56, MUTED, 1)

    _card(img, (36, 630, 1244, 850), "Retrieval Gain")
    lines = [
        f"additional predictions: {gain['additional_predictions']}",
        f"additional TP: {gain['additional_tp']}",
        f"additional FP: {gain['additional_fp']}",
        f"recall gain: {float(gain['recall_gain']):+.3f}",
        f"precision change: {float(gain['precision_change']):+.4f}",
    ]
    x = 74
    for line in lines:
        cv2.rectangle(img, (x, 690), (x + 210, 790), (50, 56, 65), -1, cv2.LINE_AA)
        head, val = line.split(": ", 1)
        _draw_text(img, head, (x + 14, 724), 0.52, MUTED, 1)
        _draw_text(img, val, (x + 14, 770), 0.82, TEXT, 2)
        x += 232

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "evaluation_summary.png"
    cv2.imwrite(str(out), img)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render report-style summary panels")
    parser.add_argument("--fusion-tracks", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fusion = render_fusion_summary(args.fusion_tracks, args.output_dir)
    evaluation = render_evaluation_summary(args.evaluation, args.output_dir)
    print("Summary panel visualization")
    print(f"- fusion summary: {fusion}")
    print(f"- evaluation summary: {evaluation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
