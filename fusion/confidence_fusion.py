from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


FINAL_STATUSES = {"accepted", "recovered", "uncertain", "rejected"}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = default
    return max(0.0, min(1.0, x))


def _index_by_track_id(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(row["track_id"]): row for row in rows}


def _source_for_status(status: str) -> str:
    if status == "accepted":
        return "mock_detector+2d_match"
    if status == "recovered":
        return "mock_detector+retrieval"
    if status == "uncertain":
        return "mock_detector+uncertain"
    return "mock_detector+rejected"


def _reason_for_status(
    *,
    final_status: str,
    matching_status: Optional[str],
    retrieval_status: Optional[str],
    track_length: int,
    avg_score_3d: float,
) -> str:
    if final_status == "accepted":
        return "strong_2d_3d_match"
    if final_status == "recovered":
        return "retrieval_recovered"
    if final_status == "uncertain":
        if matching_status == "matched_uncertain":
            return "uncertain_2d_3d_match"
        if retrieval_status == "weak_temporal_evidence":
            return "weak_temporal_retrieval_evidence"
        if track_length >= 5 and avg_score_3d >= 0.60:
            return "long_consistent_3d_track"
        return "uncertain_score"
    if track_length < 2:
        return "short_track"
    if retrieval_status in {"not_recovered_short_track", "not_recovered_no_2d_class_evidence", "not_recovered"}:
        return str(retrieval_status)
    if matching_status == "unmatched" and not retrieval_status:
        return "unmatched_no_retrieval_evidence"
    return "low_fused_score"


def _fuse_one_track(
    *,
    track: Dict[str, Any],
    match: Optional[Dict[str, Any]],
    retrieval: Optional[Dict[str, Any]],
    accepted_threshold: float,
    recovered_threshold: float,
    uncertain_threshold: float,
) -> Dict[str, Any]:
    track_id = int(track["track_id"])
    cls = str(track.get("class", ""))
    track_length = int(track.get("track_length", 0))
    avg_score_3d = _clamp01(track.get("avg_score_3d", 0.0))
    matching_status = str(match.get("status")) if match else None
    retrieval_status = str(retrieval.get("retrieval_status")) if retrieval else None

    geometry_score = avg_score_3d
    avg_score_2d = _clamp01(match.get("avg_score_2d", 0.0)) if match else 0.0
    retrieval_best = retrieval.get("best_evidence") if retrieval else None
    retrieval_score_2d = _clamp01((retrieval_best or {}).get("score_2d", 0.0)) if isinstance(retrieval_best, dict) else 0.0
    semantic_score = avg_score_2d if avg_score_2d > 0.0 else (retrieval_score_2d if retrieval_status == "recovered" else 0.0)
    match_quality = _clamp01(match.get("match_ratio_frame", 0.0)) if match else 0.0
    track_consistency = _clamp01(track_length / 10.0)
    retrieval_bonus = 0.15 if retrieval_status == "recovered" else (0.05 if retrieval_status == "weak_temporal_evidence" else 0.0)
    final_score = _clamp01(
        0.25 * geometry_score
        + 0.30 * semantic_score
        + 0.20 * track_consistency
        + 0.20 * match_quality
        + retrieval_bonus
    )

    if matching_status == "matched_high_confidence" and final_score >= accepted_threshold and track_length >= 2:
        final_status = "accepted"
    elif retrieval_status == "recovered" and final_score >= recovered_threshold and track_length >= 3:
        final_status = "recovered"
    elif (
        matching_status == "matched_uncertain"
        or retrieval_status == "weak_temporal_evidence"
        or (track_length >= 5 and avg_score_3d >= 0.60)
    ) and final_score >= uncertain_threshold:
        final_status = "uncertain"
    else:
        final_status = "rejected"

    reason = _reason_for_status(
        final_status=final_status,
        matching_status=matching_status,
        retrieval_status=retrieval_status,
        track_length=track_length,
        avg_score_3d=avg_score_3d,
    )
    return {
        "track_id": track_id,
        "class": cls,
        "track_length": track_length,
        "avg_score_3d": avg_score_3d,
        "matching_status": matching_status,
        "retrieval_status": retrieval_status,
        "match_ratio_frame": _clamp01(match.get("match_ratio_frame", 0.0)) if match else 0.0,
        "match_ratio_frame_camera": _clamp01(match.get("match_ratio_frame_camera", 0.0)) if match else 0.0,
        "avg_iou": _clamp01(match.get("avg_iou", 0.0)) if match else 0.0,
        "max_iou": _clamp01(match.get("max_iou", 0.0)) if match else 0.0,
        "avg_score_2d": avg_score_2d,
        "retrieval_best_evidence": retrieval_best if isinstance(retrieval_best, dict) else None,
        "geometry_score": geometry_score,
        "semantic_score": semantic_score,
        "track_consistency": track_consistency,
        "match_quality": match_quality,
        "retrieval_bonus": retrieval_bonus,
        "final_score": final_score,
        "final_status": final_status,
        "reason": reason,
    }


def _summary_for_tracks(fused_tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts = Counter(t["final_status"] for t in fused_tracks)
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_status: Dict[str, List[float]] = defaultdict(list)
    for track in fused_tracks:
        by_class[str(track["class"])].append(track)
        by_status[str(track["final_status"])].append(float(track["final_score"]))

    per_class: Dict[str, Dict[str, int]] = {}
    for cls, rows in sorted(by_class.items()):
        counts = Counter(row["final_status"] for row in rows)
        per_class[cls] = {
            "total": len(rows),
            "accepted": int(counts["accepted"]),
            "recovered": int(counts["recovered"]),
            "uncertain": int(counts["uncertain"]),
            "rejected": int(counts["rejected"]),
        }

    return {
        "tracks_total": len(fused_tracks),
        "accepted": int(status_counts["accepted"]),
        "recovered": int(status_counts["recovered"]),
        "uncertain": int(status_counts["uncertain"]),
        "rejected": int(status_counts["rejected"]),
        "per_class": per_class,
        "average_final_score_by_status": {
            status: float(mean(scores)) if scores else 0.0
            for status, scores in sorted(by_status.items())
        },
    }


def _build_frame_labels(
    *,
    frame_assignments_doc: Dict[str, Any],
    fused_by_track: Dict[int, Dict[str, Any]],
    include_rejected: bool,
) -> Tuple[Dict[str, Any], int]:
    frames_out: List[Dict[str, Any]] = []
    count = 0
    for frame in frame_assignments_doc.get("frames") or []:
        objects_out: List[Dict[str, Any]] = []
        for obj in frame.get("objects") or []:
            track_id = int(obj["track_id"])
            fused = fused_by_track.get(track_id)
            if not fused:
                continue
            status = str(fused["final_status"])
            if status == "rejected" and not include_rejected:
                continue
            objects_out.append(
                {
                    "track_id": track_id,
                    "det_id": str(obj.get("det_id", "")),
                    "class": str(obj.get("class", fused.get("class", ""))),
                    "box3d_lidar": [float(v) for v in obj.get("box3d_lidar", [])],
                    "score": float(fused["final_score"]),
                    "status": status,
                    "source": _source_for_status(status),
                    "components": {
                        "geometry_score": float(fused["geometry_score"]),
                        "semantic_score": float(fused["semantic_score"]),
                        "track_consistency": float(fused["track_consistency"]),
                        "match_quality": float(fused["match_quality"]),
                        "retrieval_bonus": float(fused["retrieval_bonus"]),
                    },
                }
            )
            count += 1
        frames_out.append(
            {
                "frame_id": str(frame.get("frame_id", "")),
                "timestamp": int(frame.get("timestamp", 0)),
                "objects": objects_out,
            }
        )
    return {
        "metadata": {
            "source": "confidence_fusion",
            "coordinate_frame": "lidar_like",
            "note": "Frame-level pseudo labels generated from fused track statuses",
            "include_rejected": bool(include_rejected),
        },
        "frames": frames_out,
    }, count


def run_confidence_fusion(
    *,
    tracks_path: Path,
    frame_assignments_path: Path,
    matching_path: Path,
    retrieval_path: Path,
    output_labels_path: Path,
    output_tracks_path: Path,
    include_rejected: bool = False,
    accepted_threshold: float = 0.55,
    recovered_threshold: float = 0.45,
    uncertain_threshold: float = 0.35,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    tracks_doc = _read_json(tracks_path)
    frame_assignments_doc = _read_json(frame_assignments_path)
    matching_doc = _read_json(matching_path)
    retrieval_doc = _read_json(retrieval_path)

    match_by_track = _index_by_track_id(matching_doc.get("track_matches") or [])
    retrieval_by_track = _index_by_track_id(retrieval_doc.get("track_retrieval") or [])
    fused_tracks = [
        _fuse_one_track(
            track=track,
            match=match_by_track.get(int(track["track_id"])),
            retrieval=retrieval_by_track.get(int(track["track_id"])),
            accepted_threshold=accepted_threshold,
            recovered_threshold=recovered_threshold,
            uncertain_threshold=uncertain_threshold,
        )
        for track in tracks_doc.get("tracks") or []
    ]
    summary = _summary_for_tracks(fused_tracks)
    fused_track_doc = {
        "metadata": {
            "source": "confidence_fusion",
            "tracks_input": str(tracks_path),
            "frame_assignments_input": str(frame_assignments_path),
            "matching_input": str(matching_path),
            "retrieval_input": str(retrieval_path),
            "accepted_threshold": float(accepted_threshold),
            "recovered_threshold": float(recovered_threshold),
            "uncertain_threshold": float(uncertain_threshold),
            "note": "Fuses 3D geometry score, 2D semantic evidence, track consistency, and retrieval evidence",
        },
        "summary": summary,
        "tracks": fused_tracks,
    }
    fused_by_track = {int(track["track_id"]): track for track in fused_tracks}
    labels_doc, object_count = _build_frame_labels(
        frame_assignments_doc=frame_assignments_doc,
        fused_by_track=fused_by_track,
        include_rejected=include_rejected,
    )
    labels_doc["metadata"].update(
        {
            "tracks_input": str(tracks_path),
            "frame_assignments_input": str(frame_assignments_path),
            "matching_input": str(matching_path),
            "retrieval_input": str(retrieval_path),
        }
    )
    fused_track_doc["summary"]["frame_level_pseudo_label_objects"] = object_count

    validate_fused_track_summary(fused_track_doc)
    validate_fused_pseudo_labels(labels_doc, include_rejected=include_rejected)
    _write_json(output_tracks_path, fused_track_doc)
    _write_json(output_labels_path, labels_doc)
    return fused_track_doc, labels_doc, fused_track_doc["summary"]


def validate_fused_track_summary(doc: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(doc.get("metadata"), dict):
        errors.append("fused_track_summary missing metadata")
    if not isinstance(doc.get("summary"), dict):
        errors.append("fused_track_summary missing summary")
    tracks = doc.get("tracks")
    if not isinstance(tracks, list):
        errors.append("fused_track_summary.tracks must be a list")
        return errors
    for idx, track in enumerate(tracks):
        prefix = f"tracks[{idx}]"
        for field in ("track_id", "class", "final_status", "final_score"):
            if field not in track:
                errors.append(f"{prefix} missing {field}")
        if track.get("final_status") not in FINAL_STATUSES:
            errors.append(f"{prefix}.final_status invalid")
        try:
            score = float(track.get("final_score"))
            if not (0.0 <= score <= 1.0):
                errors.append(f"{prefix}.final_score out of [0, 1]")
        except (TypeError, ValueError):
            errors.append(f"{prefix}.final_score must be numeric")
    if errors:
        raise ValueError("invalid fused track summary:\n" + "\n".join(errors[:20]))
    return errors


def validate_fused_pseudo_labels(doc: Dict[str, Any], include_rejected: bool = False) -> List[str]:
    errors: List[str] = []
    if not isinstance(doc.get("metadata"), dict):
        errors.append("fused_pseudo_labels missing metadata")
    frames = doc.get("frames")
    if not isinstance(frames, list):
        errors.append("fused_pseudo_labels.frames must be a list")
        return errors
    for frame_idx, frame in enumerate(frames):
        objects = frame.get("objects")
        if not isinstance(objects, list):
            errors.append(f"frames[{frame_idx}].objects must be a list")
            continue
        for obj_idx, obj in enumerate(objects):
            prefix = f"frames[{frame_idx}].objects[{obj_idx}]"
            for field in ("track_id", "class", "box3d_lidar", "score", "status"):
                if field not in obj:
                    errors.append(f"{prefix} missing {field}")
            if obj.get("status") == "rejected" and not include_rejected:
                errors.append(f"{prefix} rejected object included without include_rejected")
            if obj.get("status") not in FINAL_STATUSES:
                errors.append(f"{prefix}.status invalid")
            box = obj.get("box3d_lidar")
            if not isinstance(box, list) or len(box) != 7:
                errors.append(f"{prefix}.box3d_lidar must have 7 values")
            try:
                score = float(obj.get("score"))
                if not (0.0 <= score <= 1.0):
                    errors.append(f"{prefix}.score out of [0, 1]")
            except (TypeError, ValueError):
                errors.append(f"{prefix}.score must be numeric")
    if errors:
        raise ValueError("invalid fused pseudo labels:\n" + "\n".join(errors[:20]))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse 3D tracks, 2D matching, and retrieval into pseudo-label statuses")
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--frame-assignments", type=Path, required=True)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-tracks", type=Path, required=True)
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--accepted-threshold", type=float, default=0.55)
    parser.add_argument("--recovered-threshold", type=float, default=0.45)
    parser.add_argument("--uncertain-threshold", type=float, default=0.35)
    return parser.parse_args()


def _print_summary(summary: Dict[str, Any], output_labels: Path, output_tracks: Path) -> None:
    print("Confidence fusion summary")
    print(f"- total tracks: {summary['tracks_total']}")
    print(f"- accepted tracks: {summary['accepted']}")
    print(f"- recovered tracks: {summary['recovered']}")
    print(f"- uncertain tracks: {summary['uncertain']}")
    print(f"- rejected tracks: {summary['rejected']}")
    print(f"- per-class final status counts: {summary['per_class']}")
    print(f"- frame-level pseudo label object count: {summary['frame_level_pseudo_label_objects']}")
    print(f"- average final score by status: {summary['average_final_score_by_status']}")
    print(f"- output labels: {output_labels}")
    print(f"- output tracks: {output_tracks}")


def main() -> int:
    args = parse_args()
    _tracks_doc, _labels_doc, summary = run_confidence_fusion(
        tracks_path=args.tracks,
        frame_assignments_path=args.frame_assignments,
        matching_path=args.matching,
        retrieval_path=args.retrieval,
        output_labels_path=args.output_labels,
        output_tracks_path=args.output_tracks,
        include_rejected=args.include_rejected,
        accepted_threshold=args.accepted_threshold,
        recovered_threshold=args.recovered_threshold,
        uncertain_threshold=args.uncertain_threshold,
    )
    _print_summary(summary, args.output_labels, args.output_tracks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
