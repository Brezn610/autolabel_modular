from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dataset.driving_annotations import load_driving_annotations


VALID_CLASSES = ("vehicle", "pedestrian")
STATUS_GROUPS = {
    "accepted_only": {"accepted"},
    "accepted_plus_recovered": {"accepted", "recovered"},
    "accepted_recovered_uncertain": {"accepted", "recovered", "uncertain"},
    "all_non_rejected": {"accepted", "recovered", "uncertain"},
    "recovered_only": {"recovered"},
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def _center_from_box3d(box: Any) -> Optional[List[float]]:
    if not isinstance(box, list) or len(box) < 3:
        return None
    try:
        center = [float(box[0]), float(box[1]), float(box[2])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in center):
        return None
    return center


def _distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _metrics(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _index_gt_by_timestamp(gt_doc: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(frame["timestamp"]): frame for frame in gt_doc.get("frames") or []}


def _match_gt_timestamp(
    timestamp: int,
    gt_by_timestamp: Dict[int, Dict[str, Any]],
    sorted_gt_timestamps: List[int],
    tolerance_ns: int,
) -> Tuple[Optional[int], str, Optional[int]]:
    if timestamp in gt_by_timestamp:
        return timestamp, "exact", 0
    if not sorted_gt_timestamps:
        return None, "unmatched", None
    # A linear scan is fine here: 100 prediction frames and modest annotation count.
    nearest = min(sorted_gt_timestamps, key=lambda ts: abs(ts - timestamp))
    delta = abs(nearest - timestamp)
    if delta <= tolerance_ns:
        return nearest, "nearest", delta
    return None, "unmatched", delta


def _filter_gt_objects(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for obj in frame.get("objects") or []:
        cls = obj.get("normalized_class")
        if cls not in VALID_CLASSES:
            continue
        pos = obj.get("position")
        if not isinstance(pos, list) or len(pos) != 3:
            continue
        try:
            center = [float(pos[0]), float(pos[1]), float(pos[2])]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in center):
            continue
        row = dict(obj)
        row["_center"] = center
        row["_class"] = str(cls)
        out.append(row)
    return out


def _filter_predictions(frame: Dict[str, Any], allowed_statuses: set[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for obj in frame.get("objects") or []:
        if obj.get("status") not in allowed_statuses:
            continue
        cls = str(obj.get("class", ""))
        if cls not in VALID_CLASSES:
            continue
        center = _center_from_box3d(obj.get("box3d_lidar"))
        if center is None:
            continue
        try:
            score = float(obj.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        row = dict(obj)
        row["_center"] = center
        row["_class"] = cls
        row["_score"] = score
        out.append(row)
    return out


def _evaluate_frame(
    *,
    predictions: List[Dict[str, Any]],
    gt_objects: List[Dict[str, Any]],
    thresholds: Dict[str, float],
) -> Tuple[Counter[str], Dict[str, Counter[str]]]:
    totals: Counter[str] = Counter()
    per_class: Dict[str, Counter[str]] = {cls: Counter() for cls in VALID_CLASSES}

    for cls in VALID_CLASSES:
        preds_cls = [p for p in predictions if p["_class"] == cls]
        gt_cls = [g for g in gt_objects if g["_class"] == cls]
        matched_gt: set[int] = set()
        for pred in sorted(preds_cls, key=lambda p: float(p["_score"]), reverse=True):
            best_idx: Optional[int] = None
            best_dist = float("inf")
            for gt_idx, gt in enumerate(gt_cls):
                if gt_idx in matched_gt:
                    continue
                dist = _distance(pred["_center"], gt["_center"])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = gt_idx
            if best_idx is not None and best_dist <= thresholds[cls]:
                matched_gt.add(best_idx)
                totals["tp"] += 1
                per_class[cls]["tp"] += 1
            else:
                totals["fp"] += 1
                per_class[cls]["fp"] += 1
        fn = len(gt_cls) - len(matched_gt)
        totals["fn"] += fn
        per_class[cls]["fn"] += fn
        totals["predictions"] += len(preds_cls)
        per_class[cls]["predictions"] += len(preds_cls)
        totals["gt_objects"] += len(gt_cls)
        per_class[cls]["gt_objects"] += len(gt_cls)
    return totals, per_class


def _merge_per_class(dst: Dict[str, Counter[str]], src: Dict[str, Counter[str]]) -> None:
    for cls, counts in src.items():
        dst[cls].update(counts)


def _finalize_group(
    totals: Counter[str],
    per_class_counts: Dict[str, Counter[str]],
    *,
    gt_frames_matched: int,
    prediction_frames: int,
) -> Dict[str, Any]:
    out = _metrics(totals["tp"], totals["fp"], totals["fn"])
    out.update(
        {
            "predictions": int(totals["predictions"]),
            "gt_objects_considered": int(totals["gt_objects"]),
            "gt_frames_matched": int(gt_frames_matched),
            "prediction_frames": int(prediction_frames),
            "per_class": {},
        }
    )
    for cls in VALID_CLASSES:
        counts = per_class_counts[cls]
        cls_metrics = _metrics(counts["tp"], counts["fp"], counts["fn"])
        cls_metrics.update(
            {
                "predictions": int(counts["predictions"]),
                "gt_objects_considered": int(counts["gt_objects"]),
            }
        )
        out["per_class"][cls] = cls_metrics
    return out


def evaluate_status_group(
    *,
    pred_doc: Dict[str, Any],
    gt_by_timestamp: Dict[int, Dict[str, Any]],
    sorted_gt_timestamps: List[int],
    allowed_statuses: set[str],
    thresholds: Dict[str, float],
    tolerance_ns: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    totals: Counter[str] = Counter()
    per_class_counts: Dict[str, Counter[str]] = {cls: Counter() for cls in VALID_CLASSES}
    timestamp_stats: Counter[str] = Counter()
    nearest_deltas: List[int] = []
    matched_frames = 0

    for frame in pred_doc.get("frames") or []:
        pred_ts = int(frame.get("timestamp", 0))
        gt_ts, mode, delta = _match_gt_timestamp(pred_ts, gt_by_timestamp, sorted_gt_timestamps, tolerance_ns)
        timestamp_stats[mode] += 1
        if delta is not None and mode == "nearest":
            nearest_deltas.append(delta)
        predictions = _filter_predictions(frame, allowed_statuses)
        if gt_ts is None:
            totals["fp"] += len(predictions)
            totals["predictions"] += len(predictions)
            for pred in predictions:
                cls = pred["_class"]
                per_class_counts[cls]["fp"] += 1
                per_class_counts[cls]["predictions"] += 1
            continue
        matched_frames += 1
        gt_objects = _filter_gt_objects(gt_by_timestamp[gt_ts])
        frame_totals, frame_per_class = _evaluate_frame(
            predictions=predictions,
            gt_objects=gt_objects,
            thresholds=thresholds,
        )
        totals.update(frame_totals)
        _merge_per_class(per_class_counts, frame_per_class)

    result = _finalize_group(
        totals,
        per_class_counts,
        gt_frames_matched=matched_frames,
        prediction_frames=len(pred_doc.get("frames") or []),
    )
    matching_info = {
        "timestamp_matching": {
            "exact": int(timestamp_stats["exact"]),
            "nearest": int(timestamp_stats["nearest"]),
            "unmatched": int(timestamp_stats["unmatched"]),
            "tolerance_ns": int(tolerance_ns),
            "nearest_delta_min_ns": int(min(nearest_deltas)) if nearest_deltas else None,
            "nearest_delta_max_ns": int(max(nearest_deltas)) if nearest_deltas else None,
        }
    }
    return result, matching_info


def _retrieval_gain(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "additional_predictions": int(after["predictions"] - before["predictions"]),
        "additional_tp": int(after["tp"] - before["tp"]),
        "additional_fp": int(after["fp"] - before["fp"]),
        "recall_before": float(before["recall"]),
        "recall_after": float(after["recall"]),
        "recall_gain": float(after["recall"] - before["recall"]),
        "precision_before": float(before["precision"]),
        "precision_after": float(after["precision"]),
        "precision_change": float(after["precision"] - before["precision"]),
    }


def _load_track_summary_if_available(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    doc = _read_json(path)
    rows = doc.get("tracks") or []
    status_counts = Counter(row.get("final_status") for row in rows)
    scores_by_status: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        try:
            scores_by_status[str(row.get("final_status"))].append(float(row.get("final_score", 0.0)))
        except (TypeError, ValueError):
            continue
    return {
        "track_counts_by_status": dict(status_counts),
        "average_score_by_status": {
            status: float(mean(scores)) if scores else 0.0
            for status, scores in sorted(scores_by_status.items())
        },
    }


def _write_ablation_csv(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "method",
        "predictions",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "vehicle_precision",
        "vehicle_recall",
        "pedestrian_precision",
        "pedestrian_recall",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for method in ("accepted_only", "accepted_plus_recovered", "accepted_recovered_uncertain", "recovered_only"):
            row = summary[method]
            writer.writerow(
                {
                    "method": method,
                    "predictions": row["predictions"],
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "vehicle_precision": row["per_class"]["vehicle"]["precision"],
                    "vehicle_recall": row["per_class"]["vehicle"]["recall"],
                    "pedestrian_precision": row["per_class"]["pedestrian"]["precision"],
                    "pedestrian_recall": row["per_class"]["pedestrian"]["recall"],
                }
            )


def evaluate_pseudo_labels(
    *,
    predictions_path: Path,
    gt_path: Path,
    output_json_path: Path,
    output_csv_path: Path,
    vehicle_threshold: float,
    pedestrian_threshold: float,
    timestamp_tolerance_ns: int,
    track_summary_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pred_doc = _read_json(predictions_path)
    gt_doc = load_driving_annotations(gt_path, source_frame="unknown")
    gt_by_timestamp = _index_gt_by_timestamp(gt_doc)
    sorted_gt_timestamps = sorted(gt_by_timestamp)
    thresholds = {"vehicle": float(vehicle_threshold), "pedestrian": float(pedestrian_threshold)}

    summary: Dict[str, Any] = {}
    timestamp_matching_info: Dict[str, Any] = {}
    for name, statuses in STATUS_GROUPS.items():
        metrics, matching_info = evaluate_status_group(
            pred_doc=pred_doc,
            gt_by_timestamp=gt_by_timestamp,
            sorted_gt_timestamps=sorted_gt_timestamps,
            allowed_statuses=set(statuses),
            thresholds=thresholds,
            tolerance_ns=int(timestamp_tolerance_ns),
        )
        summary[name] = metrics
        if not timestamp_matching_info:
            timestamp_matching_info = matching_info

    retrieval_gain = _retrieval_gain(summary["accepted_only"], summary["accepted_plus_recovered"])
    track_summary = _load_track_summary_if_available(track_summary_path)
    out_doc = {
        "metadata": {
            "source": "eval_pseudo_labels",
            "prediction_input": str(predictions_path),
            "gt_input": str(gt_path),
            "track_summary_input": str(track_summary_path) if track_summary_path else None,
            "metric": "center_distance",
            "thresholds": thresholds,
            "timestamp_tolerance_ns": int(timestamp_tolerance_ns),
            "coordinate_frame_note": "Predictions are lidar_like; GT annotation coordinate frame was inferred as lidar-like during earlier BEV diagnostics but is not explicitly declared in annotations.json.",
            "note": "Evaluation for mock-detector pipeline prototype",
        },
        "summary": summary,
        "retrieval_gain": retrieval_gain,
        "timestamp_matching": timestamp_matching_info.get("timestamp_matching", {}),
        "track_summary": track_summary,
        "per_frame_optional": [],
    }
    validate_evaluation_doc(out_doc)
    _write_json(output_json_path, out_doc)
    _write_ablation_csv(output_csv_path, summary)
    return out_doc, summary


def validate_evaluation_doc(doc: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(doc.get("metadata"), dict):
        errors.append("missing metadata")
    if not isinstance(doc.get("summary"), dict):
        errors.append("missing summary")
    for name in STATUS_GROUPS:
        row = (doc.get("summary") or {}).get(name)
        if not isinstance(row, dict):
            errors.append(f"summary missing {name}")
            continue
        for field in ("tp", "fp", "fn", "precision", "recall", "f1", "predictions", "gt_objects_considered"):
            if field not in row:
                errors.append(f"summary.{name} missing {field}")
    if errors:
        raise ValueError("invalid evaluation output:\n" + "\n".join(errors[:20]))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fused pseudo labels against normalized DrivIng annotations")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--vehicle-threshold", type=float, default=2.0)
    parser.add_argument("--pedestrian-threshold", type=float, default=1.0)
    parser.add_argument("--timestamp-tolerance-ns", type=int, default=50_000_000)
    parser.add_argument("--track-summary", type=Path, default=Path("outputs/fusion/fused_track_summary_100f.json"))
    return parser.parse_args()


def _fmt_metrics(row: Dict[str, Any]) -> str:
    return (
        f"pred={row['predictions']} TP={row['tp']} FP={row['fp']} FN={row['fn']} "
        f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}"
    )


def _print_summary(doc: Dict[str, Any], output_json: Path, output_csv: Path) -> None:
    summary = doc["summary"]
    print("Pseudo-label evaluation summary")
    print(f"- prediction frames: {summary['accepted_recovered_uncertain']['prediction_frames']}")
    print(f"- GT frames matched: {summary['accepted_recovered_uncertain']['gt_frames_matched']}")
    print(f"- GT objects considered: {summary['accepted_recovered_uncertain']['gt_objects_considered']}")
    print(f"- timestamp matching: {doc['timestamp_matching']}")
    print(f"- accepted_only: {_fmt_metrics(summary['accepted_only'])}")
    print(f"- accepted_plus_recovered: {_fmt_metrics(summary['accepted_plus_recovered'])}")
    print(f"- accepted_recovered_uncertain: {_fmt_metrics(summary['accepted_recovered_uncertain'])}")
    gain = doc["retrieval_gain"]
    print(
        "- retrieval recall gain: "
        f"{gain['recall_gain']:.3f} "
        f"(additional TP={gain['additional_tp']}, additional FP={gain['additional_fp']})"
    )
    print(f"- output JSON: {output_json}")
    print(f"- output CSV: {output_csv}")


def main() -> int:
    args = parse_args()
    doc, _summary = evaluate_pseudo_labels(
        predictions_path=args.predictions,
        gt_path=args.gt,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
        vehicle_threshold=args.vehicle_threshold,
        pedestrian_threshold=args.pedestrian_threshold,
        timestamp_tolerance_ns=args.timestamp_tolerance_ns,
        track_summary_path=args.track_summary,
    )
    _print_summary(doc, args.output_json, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
