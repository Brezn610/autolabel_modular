from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from detector3d.detector_provider import load_detections_json
from tracking.simple_3d_tracker import run_simple_tracker


VEHICLE_THRESHOLDS = [2.5, 4.0, 6.0, 8.0, 10.0]
PEDESTRIAN_THRESHOLDS = [1.0, 1.5, 2.0, 3.0]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    vals = sorted(float(v) for v in values)

    def pct(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        k = (len(vals) - 1) * p
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - k) + vals[hi] * (k - lo)

    return {
        "count": len(vals),
        "min": vals[0],
        "median": median(vals),
        "mean": mean(vals),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": vals[-1],
    }


def _center_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def _det_debug_index(det_doc: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    by_det_id: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    frames = sorted(det_doc["frames"], key=lambda f: (int(f["timestamp"]), str(f["frame_id"])))
    for frame_index, frame in enumerate(frames):
        for det in frame.get("boxes3d") or []:
            row = {
                "frame_index": frame_index,
                "frame_id": str(frame["frame_id"]),
                "timestamp": int(frame["timestamp"]),
                "det_id": str(det["det_id"]),
                "class": str(det["class"]),
                "box3d_lidar": [float(x) for x in det["box3d_lidar"]],
                "score_3d": float(det["score_3d"]),
                "mock_from_gt": bool(det.get("mock_from_gt", False)),
                "original_track_id": det.get("original_track_id"),
                "gt_id": det.get("gt_id"),
            }
            by_det_id[row["det_id"]] = row
            rows.append(row)
    return by_det_id, rows


def frame_continuity(det_doc: Dict[str, Any]) -> Dict[str, Any]:
    frames = sorted(det_doc["frames"], key=lambda f: (int(f["timestamp"]), str(f["frame_id"])))
    timestamps = [int(f["timestamp"]) for f in frames]
    deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    stats = _stats([float(d) for d in deltas])
    median_delta = stats["median"] or 0
    large_gaps = []
    if median_delta:
        for i, delta in enumerate(deltas, start=1):
            if delta > 1.5 * float(median_delta):
                large_gaps.append(
                    {
                        "prev_frame_id": str(frames[i - 1]["frame_id"]),
                        "frame_id": str(frames[i]["frame_id"]),
                        "prev_timestamp": timestamps[i - 1],
                        "timestamp": timestamps[i],
                        "delta": int(delta),
                    }
                )
    return {
        "num_frames": len(frames),
        "timestamp_delta_stats": stats,
        "large_gaps": large_gaps[:50],
        "large_gap_count": len(large_gaps),
    }


def oracle_gt_continuity(det_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_gt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in det_rows:
        if row["mock_from_gt"] and row["original_track_id"] is not None:
            by_gt[str(row["original_track_id"])].append(row)
    lengths = [len(v) for v in by_gt.values()]
    return {
        "unique_original_gt_tracks": len(by_gt),
        "avg_detections_per_original_track_id": float(mean(lengths)) if lengths else 0.0,
        "median_detections_per_original_track_id": float(median(lengths)) if lengths else 0.0,
        "longest_original_track_id_length": max(lengths) if lengths else 0,
        "original_gt_tracks_length_ge_3": sum(1 for x in lengths if x >= 3),
    }


def _assignment_pairs(
    frame_doc: Dict[str, Any],
    det_index: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, set[int]], Dict[int, set[str]]]:
    gt_to_pred: Dict[str, set[int]] = defaultdict(set)
    pred_to_gt: Dict[int, set[str]] = defaultdict(set)
    for frame in frame_doc.get("frames") or []:
        for obj in frame.get("objects") or []:
            det_id = str(obj.get("det_id"))
            det = det_index.get(det_id)
            if not det or not det["mock_from_gt"] or det["original_track_id"] is None:
                continue
            gt_id = str(det["original_track_id"])
            pred_id = int(obj["track_id"])
            gt_to_pred[gt_id].add(pred_id)
            pred_to_gt[pred_id].add(gt_id)
    return gt_to_pred, pred_to_gt


def fragmentation_summary(frame_doc: Dict[str, Any], det_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    gt_to_pred, pred_to_gt = _assignment_pairs(frame_doc, det_index)
    frags = [len(v) for v in gt_to_pred.values()]
    mixed_pred = {str(k): sorted(v) for k, v in pred_to_gt.items() if len(v) > 1}
    top = sorted(
        (
            {
                "original_track_id": gt,
                "predicted_track_count": len(preds),
                "predicted_track_ids": sorted(preds),
            }
            for gt, preds in gt_to_pred.items()
        ),
        key=lambda x: x["predicted_track_count"],
        reverse=True,
    )[:20]
    return {
        "original_gt_tracks_with_assignments": len(gt_to_pred),
        "avg_fragmentation_count": float(mean(frags)) if frags else 0.0,
        "median_fragmentation_count": float(median(frags)) if frags else 0.0,
        "max_fragmentation_count": max(frags) if frags else 0,
        "top_20_most_fragmented_original_tracks": top,
        "potential_id_switch_tracks": len(mixed_pred),
        "potential_id_switch_examples": dict(list(mixed_pred.items())[:20]),
    }


def same_gt_consecutive_distances(det_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_gt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in det_rows:
        if row["mock_from_gt"] and row["original_track_id"] is not None:
            by_gt[str(row["original_track_id"])].append(row)
    by_class: Dict[str, List[float]] = defaultdict(list)
    all_distances: List[float] = []
    for rows in by_gt.values():
        rows.sort(key=lambda r: (int(r["frame_index"]), int(r["timestamp"])))
        for prev, cur in zip(rows, rows[1:]):
            if int(cur["frame_index"]) != int(prev["frame_index"]) + 1:
                continue
            dist = _center_distance(prev["box3d_lidar"], cur["box3d_lidar"])
            cls = str(cur["class"])
            by_class[cls].append(dist)
            all_distances.append(dist)
    out = {"all": _stats(all_distances)}
    for cls in ("vehicle", "pedestrian"):
        out[cls] = _stats(by_class.get(cls, []))
    return out


def _tracker_doc_to_frame_doc(track_doc: Dict[str, Any]) -> Dict[str, Any]:
    # Unused currently, kept for future compatibility if only track output is supplied.
    frames: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for track in track_doc.get("tracks") or []:
        for obs in track.get("frames") or []:
            key = (str(obs["frame_id"]), int(obs["timestamp"]))
            frames[key].append(
                {
                    "track_id": int(track["track_id"]),
                    "det_id": str(obs["det_id"]),
                    "class": str(track["class"]),
                    "box3d_lidar": obs["box3d_lidar"],
                    "score_3d": obs["score_3d"],
                }
            )
    return {
        "frames": [
            {"frame_id": fid, "timestamp": ts, "objects": objs}
            for (fid, ts), objs in sorted(frames.items(), key=lambda x: (x[0][1], x[0][0]))
        ]
    }


def run_diagnostics(detections: Path, tracks: Path, frame_assignments: Path) -> Dict[str, Any]:
    det_doc = load_detections_json(detections)
    track_doc = _read_json(tracks)
    frame_doc = _read_json(frame_assignments)
    det_index, det_rows = _det_debug_index(det_doc)
    return {
        "inputs": {
            "detections": str(detections),
            "tracks": str(tracks),
            "frame_assignments": str(frame_assignments),
        },
        "frame_continuity": frame_continuity(det_doc),
        "oracle_gt_track_continuity": oracle_gt_continuity(det_rows),
        "tracker_summary": (track_doc.get("metadata") or {}).get("summary", {}),
        "tracker_fragmentation": fragmentation_summary(frame_doc, det_index),
        "same_gt_consecutive_distance_stats": same_gt_consecutive_distances(det_rows),
        "recommendation": {},
    }


def _summary_from_track_doc(track_doc: Dict[str, Any]) -> Dict[str, Any]:
    return (track_doc.get("metadata") or {}).get("summary", {})


def run_threshold_sweep(detections: Path, max_age: int = 2) -> Dict[str, Any]:
    det_doc = load_detections_json(detections)
    det_index, det_rows = _det_debug_index(det_doc)
    distance_stats = same_gt_consecutive_distances(det_rows)
    rows: List[Dict[str, Any]] = []
    for vehicle_dist in VEHICLE_THRESHOLDS:
        for pedestrian_dist in PEDESTRIAN_THRESHOLDS:
            track_doc, frame_doc, summary = run_simple_tracker(
                detections,
                vehicle_dist=vehicle_dist,
                pedestrian_dist=pedestrian_dist,
                max_age=max_age,
            )
            frag = fragmentation_summary(frame_doc, det_index)
            rows.append(
                {
                    "vehicle_dist": vehicle_dist,
                    "pedestrian_dist": pedestrian_dist,
                    **summary,
                    "avg_fragmentation_count": frag["avg_fragmentation_count"],
                    "median_fragmentation_count": frag["median_fragmentation_count"],
                    "max_fragmentation_count": frag["max_fragmentation_count"],
                    "potential_id_switch_tracks": frag["potential_id_switch_tracks"],
                }
            )
    best = _choose_sweep_recommendation(rows, distance_stats)
    return {
        "inputs": {"detections": str(detections), "max_age": int(max_age)},
        "same_gt_consecutive_distance_stats": distance_stats,
        "sweep": rows,
        "recommendation": best,
    }


def _choose_sweep_recommendation(rows: List[Dict[str, Any]], distance_stats: Dict[str, Any]) -> Dict[str, Any]:
    if not rows:
        return {}
    vehicle_p90 = distance_stats.get("vehicle", {}).get("p90")
    vehicle_p95 = distance_stats.get("vehicle", {}).get("p95")
    pedestrian_p90 = distance_stats.get("pedestrian", {}).get("p90")
    vehicle_candidates = sorted({float(r["vehicle_dist"]) for r in rows})
    pedestrian_candidates = sorted({float(r["pedestrian_dist"]) for r in rows})

    def pick_at_least(candidates: List[float], target: Optional[float], fallback: float) -> float:
        if target is None:
            return fallback
        for value in candidates:
            if value >= target:
                return value
        return candidates[-1]

    # Evidence-first recommendation: use the smallest swept threshold that covers
    # about the p95 same-GT inter-frame motion. Larger thresholds may reduce
    # fragmentation further but are more likely to hide over-merging in greedy
    # matching.
    target_vehicle = float(vehicle_p95) * 1.05 if vehicle_p95 is not None else None
    target_pedestrian = float(pedestrian_p90) * 1.10 if pedestrian_p90 is not None else None
    selected_vehicle = pick_at_least(vehicle_candidates, target_vehicle, 6.0)
    selected_pedestrian = pick_at_least(pedestrian_candidates, target_pedestrian, 1.5)
    selected_rows = [
        r for r in rows
        if float(r["vehicle_dist"]) == selected_vehicle and float(r["pedestrian_dist"]) == selected_pedestrian
    ]
    if selected_rows:
        best = selected_rows[0]
    else:
        best = min(
            rows,
            key=lambda r: (
                abs(float(r["vehicle_dist"]) - selected_vehicle),
                abs(float(r["pedestrian_dist"]) - selected_pedestrian),
            ),
        )
    evidence = []
    if vehicle_p90 is not None:
        evidence.append(f"vehicle same-GT p90 distance is {vehicle_p90:.2f}m")
    if vehicle_p95 is not None:
        evidence.append(f"vehicle same-GT p95 distance is {vehicle_p95:.2f}m")
    if pedestrian_p90 is not None:
        evidence.append(f"pedestrian same-GT p90 distance is {pedestrian_p90:.2f}m")
    else:
        evidence.append("no consecutive pedestrian same-GT pairs were available; pedestrian threshold is weakly constrained")
    return {
        "vehicle_dist": best["vehicle_dist"],
        "pedestrian_dist": best["pedestrian_dist"],
        "max_age": best.get("max_age", 2),
        "reason": (
            "Selected from sweep using same-GT distance statistics first: smallest swept threshold "
            "covering roughly p95 vehicle motion, with pedestrian threshold left conservative because "
            "no consecutive pedestrian evidence was available."
        ),
        "evidence": evidence,
        "best_row": best,
    }


def add_recommendation(doc: Dict[str, Any], sweep_doc: Optional[Dict[str, Any]] = None) -> None:
    dist = doc["same_gt_consecutive_distance_stats"]
    rec: Dict[str, Any] = {}
    vehicle_p90 = dist.get("vehicle", {}).get("p90")
    pedestrian_p90 = dist.get("pedestrian", {}).get("p90")
    if vehicle_p90 is not None:
        rec["vehicle_dist_evidence"] = f"vehicle same-GT p90 center distance = {vehicle_p90:.2f}m"
        rec["vehicle_dist_initial_suggestion"] = max(2.5, round(float(vehicle_p90) * 1.1, 1))
    if pedestrian_p90 is not None:
        rec["pedestrian_dist_evidence"] = f"pedestrian same-GT p90 center distance = {pedestrian_p90:.2f}m"
        rec["pedestrian_dist_initial_suggestion"] = max(1.0, round(float(pedestrian_p90) * 1.1, 1))
    frame_gaps = doc["frame_continuity"].get("large_gap_count", 0)
    if frame_gaps:
        rec["frame_gap_warning"] = f"{frame_gaps} timestamp gaps exceed 1.5x median frame delta"
    oracle = doc["oracle_gt_track_continuity"]
    if oracle.get("original_gt_tracks_length_ge_3", 0) == 0:
        rec["short_track_warning"] = "original GT tracks mostly appear fewer than 3 times, so short tracker tracks are expected"
    if sweep_doc:
        rec["sweep_recommendation"] = sweep_doc.get("recommendation", {})
    doc["recommendation"] = rec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostics for simple 3D tracking outputs")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, default=Path(""))
    parser.add_argument("--frame-assignments", type=Path, default=Path(""))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--max-age", type=int, default=2)
    return parser.parse_args()


def _print_diagnostics(doc: Dict[str, Any]) -> None:
    print("Tracking diagnostics summary")
    fc = doc.get("frame_continuity", {})
    print(f"- frames: {fc.get('num_frames')}")
    print(f"- timestamp delta stats: {fc.get('timestamp_delta_stats')}")
    oracle = doc.get("oracle_gt_track_continuity", {})
    print(f"- unique original GT tracks: {oracle.get('unique_original_gt_tracks')}")
    print(f"- avg detections/original GT: {oracle.get('avg_detections_per_original_track_id'):.2f}")
    print(f"- median detections/original GT: {oracle.get('median_detections_per_original_track_id'):.2f}")
    frag = doc.get("tracker_fragmentation", {})
    print(f"- avg fragmentation: {frag.get('avg_fragmentation_count'):.2f}")
    print(f"- median fragmentation: {frag.get('median_fragmentation_count'):.2f}")
    print(f"- max fragmentation: {frag.get('max_fragmentation_count')}")
    dist = doc.get("same_gt_consecutive_distance_stats", {})
    print(f"- vehicle same-GT distance stats: {dist.get('vehicle')}")
    print(f"- pedestrian same-GT distance stats: {dist.get('pedestrian')}")
    print(f"- recommendation: {doc.get('recommendation')}")


def _print_sweep(doc: Dict[str, Any]) -> None:
    print("Tracking threshold sweep summary")
    print(f"- combinations: {len(doc.get('sweep', []))}")
    print(f"- vehicle distance stats: {doc.get('same_gt_consecutive_distance_stats', {}).get('vehicle')}")
    print(f"- pedestrian distance stats: {doc.get('same_gt_consecutive_distance_stats', {}).get('pedestrian')}")
    print(f"- recommendation: {doc.get('recommendation')}")
    for row in doc.get("sweep", [])[:5]:
        print(f"  row: {row}")


def main() -> int:
    args = parse_args()
    if args.sweep:
        doc = run_threshold_sweep(args.detections, max_age=args.max_age)
        _write_json(args.output, doc)
        _print_sweep(doc)
        print(f"Wrote sweep diagnostics: {args.output}")
        return 0

    if not args.tracks or not args.tracks.is_file():
        raise FileNotFoundError("--tracks is required unless --sweep is set")
    if not args.frame_assignments or not args.frame_assignments.is_file():
        raise FileNotFoundError("--frame-assignments is required unless --sweep is set")
    doc = run_diagnostics(args.detections, args.tracks, args.frame_assignments)
    add_recommendation(doc)
    _write_json(args.output, doc)
    _print_diagnostics(doc)
    print(f"Wrote tracking diagnostics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
