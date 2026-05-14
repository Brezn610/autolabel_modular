from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import cv2

from dataset.annotation_projection_probe import _camera_image_path, _load_timesync
from fusion.match_2d_3d import (
    _class_compatible,
    _camera_names,
    _frames_by_id,
    _infer_image_sizes,
    _prepare_detections_for_frame,
    clip_box2d,
    iou_2d,
)


RETRIEVAL_STATUSES = {
    "recovered",
    "weak_temporal_evidence",
    "not_recovered",
    "not_recovered_short_track",
    "not_recovered_no_2d_class_evidence",
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def _track_projection_index(projection_doc: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    tracks: Dict[int, Dict[str, Any]] = {}
    for frame in projection_doc.get("frames") or []:
        frame_id = str(frame.get("frame_id"))
        timestamp = int(frame.get("timestamp", 0))
        for obj in frame.get("objects") or []:
            track_id = int(obj["track_id"])
            row = tracks.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "class": str(obj.get("class", "")),
                    "scores_3d": [],
                    "observations": [],
                },
            )
            row["scores_3d"].append(float(obj.get("score_3d", 0.0)))
            row["observations"].append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "det_id": str(obj.get("det_id", "")),
                    "class": str(obj.get("class", "")),
                    "camera_projections": obj.get("camera_projections") or {},
                }
            )
    return tracks


def _candidate_tracks(matching_doc: Dict[str, Any], include_uncertain: bool) -> List[Dict[str, Any]]:
    statuses = {"unmatched"}
    if include_uncertain:
        statuses.add("matched_uncertain")
    return [track for track in matching_doc.get("track_matches") or [] if track.get("status") in statuses]


def _has_compatible_2d_class(
    *,
    track_class: str,
    detections_doc: Dict[str, Any],
    min_score_2d: float,
) -> bool:
    for frame in detections_doc.get("frames") or []:
        for detections in (frame.get("cameras") or {}).values():
            for det in detections or []:
                try:
                    score = float(det.get("score_2d", 0.0))
                except (TypeError, ValueError):
                    continue
                if score >= min_score_2d and _class_compatible(track_class, det.get("class"), det.get("source_label", "")):
                    return True
    return False


def _evidence_from_match(
    *,
    frame_id: str,
    camera: str,
    det: Dict[str, Any],
    iou: float,
    projected_box2d: List[float],
) -> Dict[str, Any]:
    return {
        "frame_id": frame_id,
        "camera": camera,
        "matched_2d_id": str(det.get("det2d_id", "")),
        "iou": float(iou),
        "score_2d": float(det.get("score_2d", 0.0)),
        "source_label": str(det.get("source_label", "")),
        "projected_box2d": projected_box2d,
        "box2d": [float(v) for v in det.get("_box2d_clipped", det.get("box2d", []))],
    }


def _search_track_evidence(
    *,
    track_info: Dict[str, Any],
    detections_by_frame: Dict[str, Dict[str, Any]],
    image_sizes: Dict[str, Tuple[int, int]],
    cameras: List[str],
    relaxed_iou_threshold: float,
    retrieval_min_score_2d: float,
    box_counts: Counter[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    evidence: List[Dict[str, Any]] = []
    weak_evidence: List[Dict[str, Any]] = []
    track_class = str(track_info.get("class", ""))

    for obs in track_info.get("observations") or []:
        frame_id = str(obs["frame_id"])
        det_frame = detections_by_frame.get(frame_id, {"cameras": {}})
        clean_dets_by_camera = _prepare_detections_for_frame(
            frame=det_frame,
            image_sizes=image_sizes,
            min_score_2d=retrieval_min_score_2d,
            summary=box_counts,
        )
        projections = obs.get("camera_projections") or {}
        for camera in cameras:
            proj = projections.get(camera) or {}
            if not proj.get("in_fov"):
                continue
            width, height = image_sizes.get(camera, (None, None))
            projected = clip_box2d(proj.get("projected_box2d"), width, height)
            if projected is None:
                box_counts["box_invalid_count"] += 1
                continue
            best_det: Optional[Dict[str, Any]] = None
            best_iou = 0.0
            for det in clean_dets_by_camera.get(camera) or []:
                if not _class_compatible(track_class, det.get("class"), det.get("source_label", "")):
                    continue
                iou = iou_2d(projected, det.get("_box2d_clipped"), width, height)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
            if best_det is None:
                continue
            item = _evidence_from_match(
                frame_id=frame_id,
                camera=camera,
                det=best_det,
                iou=best_iou,
                projected_box2d=projected,
            )
            if best_iou >= relaxed_iou_threshold:
                evidence.append(item)
            elif best_iou >= 0.10:
                weak_evidence.append(item)
    return evidence, weak_evidence


def run_retrieval(
    *,
    matching_path: Path,
    projection_path: Path,
    detections_2d_path: Path,
    output_path: Path,
    relaxed_iou_threshold: float = 0.15,
    retrieval_min_score_2d: float = 0.50,
    include_uncertain: bool = False,
    data_root: Optional[Path] = None,
    debug_images: bool = False,
    debug_dir: Path = Path("outputs/debug/retrieval_probe"),
    max_debug_examples: int = 30,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    matching_doc = _read_json(matching_path)
    projection_doc = _read_json(projection_path)
    detections_doc = _read_json(detections_2d_path)
    cameras = _camera_names(projection_doc, detections_doc)
    image_sizes, warnings, timesync = _infer_image_sizes(
        projection_doc=projection_doc,
        data_root=data_root,
        cameras=cameras,
    )
    detections_by_frame = _frames_by_id(detections_doc)
    projection_by_track = _track_projection_index(projection_doc)
    candidates = _candidate_tracks(matching_doc, include_uncertain=include_uncertain)

    class_has_2d = {
        cls: _has_compatible_2d_class(
            track_class=cls,
            detections_doc=detections_doc,
            min_score_2d=retrieval_min_score_2d,
        )
        for cls in ("vehicle", "pedestrian")
    }

    box_counts: Counter[str] = Counter()
    results: List[Dict[str, Any]] = []
    recovered_debug: List[Dict[str, Any]] = []

    for track in candidates:
        track_id = int(track["track_id"])
        original_status = str(track.get("status", ""))
        info = projection_by_track.get(track_id, {"class": str(track.get("class", "")), "scores_3d": [], "observations": []})
        track_class = str(info.get("class") or track.get("class", ""))
        track_length = len(info.get("observations") or [])
        scores = [float(v) for v in info.get("scores_3d") or []]
        avg_score_3d = float(mean(scores)) if scores else 0.0

        evidence: List[Dict[str, Any]] = []
        weak_evidence: List[Dict[str, Any]] = []
        if track_length >= 3 and class_has_2d.get(track_class, False) and avg_score_3d >= 0.45:
            evidence, weak_evidence = _search_track_evidence(
                track_info=info,
                detections_by_frame=detections_by_frame,
                image_sizes=image_sizes,
                cameras=cameras,
                relaxed_iou_threshold=relaxed_iou_threshold,
                retrieval_min_score_2d=retrieval_min_score_2d,
                box_counts=box_counts,
            )

        best_evidence = max(evidence, key=lambda item: (float(item["iou"]), float(item["score_2d"])), default=None)
        if track_length < 3:
            status = "not_recovered_short_track"
        elif not class_has_2d.get(track_class, False):
            status = "not_recovered_no_2d_class_evidence"
        elif avg_score_3d < 0.45:
            status = "not_recovered"
        elif evidence:
            status = "recovered"
        elif weak_evidence:
            status = "weak_temporal_evidence"
            best_evidence = max(weak_evidence, key=lambda item: (float(item["iou"]), float(item["score_2d"])), default=None)
        else:
            status = "not_recovered"

        row = {
            "track_id": track_id,
            "class": track_class,
            "original_status": original_status,
            "retrieval_status": status,
            "track_length": track_length,
            "avg_score_3d": avg_score_3d,
            "best_evidence": best_evidence,
            "evidence_count": len(evidence),
            "weak_evidence_count": len(weak_evidence),
        }
        results.append(row)
        if status == "recovered" and best_evidence is not None:
            recovered_debug.append(row)

    status_counts = Counter(row["retrieval_status"] for row in results)
    recovered = [row for row in results if row["retrieval_status"] == "recovered" and row.get("best_evidence")]
    recovered_by_class = Counter(row["class"] for row in recovered)
    best_ious = [float(row["best_evidence"]["iou"]) for row in recovered]
    best_scores = [float(row["best_evidence"]["score_2d"]) for row in recovered]
    summary = {
        "candidate_tracks": len(results),
        "recovered_tracks": int(status_counts["recovered"]),
        "weak_temporal_evidence_tracks": int(status_counts["weak_temporal_evidence"]),
        "not_recovered_tracks": int(status_counts["not_recovered"]),
        "not_recovered_short_track": int(status_counts["not_recovered_short_track"]),
        "not_recovered_no_2d_class_evidence": int(status_counts["not_recovered_no_2d_class_evidence"]),
        "recovered_by_class": dict(recovered_by_class),
        "best_iou_avg": float(mean(best_ious)) if best_ious else 0.0,
        "best_score_2d_avg": float(mean(best_scores)) if best_scores else 0.0,
        "box_invalid_count": int(box_counts["box_invalid_count"] + box_counts["detections_2d_invalid"]),
        "detections_2d_processed": int(box_counts["detections_2d_processed"]),
        "detections_2d_clipped": int(box_counts["detections_2d_clipped"]),
        "warnings": warnings,
        "debug_image_dir": str(debug_dir) if debug_images else "",
    }
    out_doc = {
        "metadata": {
            "source": "retrieval",
            "matching_input": str(matching_path),
            "projection_input": str(projection_path),
            "detections_2d_input": str(detections_2d_path),
            "relaxed_iou_threshold": float(relaxed_iou_threshold),
            "retrieval_min_score_2d": float(retrieval_min_score_2d),
            "include_uncertain": bool(include_uncertain),
            "note": "Retrieval searches relaxed 2D evidence for unmatched 3D tracks",
            "warnings": warnings,
        },
        "summary": summary,
        "track_retrieval": results,
    }
    validate_retrieval_doc(out_doc)
    _write_json(output_path, out_doc)

    if debug_images:
        _write_debug_images(
            recovered=recovered_debug[: int(max_debug_examples)],
            data_root=data_root or Path(projection_doc.get("metadata", {}).get("data_root", "")),
            timesync=timesync,
            debug_dir=debug_dir,
            image_sizes=image_sizes,
        )
    return out_doc, summary


def _write_debug_images(
    *,
    recovered: List[Dict[str, Any]],
    data_root: Path,
    timesync: Dict[int, Dict[str, str]],
    debug_dir: Path,
    image_sizes: Dict[str, Tuple[int, int]],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts_by_frame_id = {f"{idx:06d}": ts for idx, ts in enumerate(sorted(timesync))}
    for idx, row in enumerate(recovered):
        ev = row.get("best_evidence") or {}
        frame_id = str(ev.get("frame_id", ""))
        camera = str(ev.get("camera", ""))
        timestamp = ts_by_frame_id.get(frame_id)
        if timestamp is None:
            continue
        image_name = (timesync.get(timestamp) or {}).get(camera)
        if not image_name:
            continue
        image_path = _camera_image_path(data_root, camera, image_name)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        if camera in image_sizes:
            w, h = image_sizes[camera]
            if image.shape[1] != w or image.shape[0] != h:
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        canvas = image.copy()
        proj_box = clip_box2d(ev.get("projected_box2d"), image.shape[1], image.shape[0])
        det_box = clip_box2d(ev.get("box2d"), image.shape[1], image.shape[0])
        if det_box:
            _draw_rect(canvas, det_box, (255, 160, 0), f"2D {float(ev.get('score_2d', 0.0)):.2f}", 1)
        if proj_box:
            label = f"t{row['track_id']} {row['retrieval_status']} IoU {float(ev.get('iou', 0.0)):.2f} s {float(ev.get('score_2d', 0.0)):.2f}"
            _draw_rect(canvas, proj_box, (0, 255, 0), label, 2)
        out_name = f"{idx:03d}__track_{row['track_id']}__frame_{frame_id}__{camera}.jpg"
        cv2.imwrite(str(debug_dir / out_name), canvas)


def _draw_rect(image: Any, box: List[float], color: Tuple[int, int, int], label: str, thickness: int) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.putText(image, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def validate_retrieval_doc(doc: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(doc.get("metadata"), dict):
        errors.append("metadata must exist")
    rows = doc.get("track_retrieval")
    if not isinstance(rows, list):
        errors.append("track_retrieval must exist and be a list")
        return errors
    for idx, row in enumerate(rows):
        prefix = f"track_retrieval[{idx}]"
        for field in ("track_id", "class", "original_status", "retrieval_status"):
            if field not in row:
                errors.append(f"{prefix} missing {field}")
        if row.get("retrieval_status") not in RETRIEVAL_STATUSES:
            errors.append(f"{prefix}.retrieval_status is invalid")
        ev = row.get("best_evidence")
        if row.get("retrieval_status") == "recovered":
            if not isinstance(ev, dict):
                errors.append(f"{prefix}.best_evidence must exist for recovered tracks")
                continue
            for field in ("frame_id", "camera", "iou", "score_2d"):
                if field not in ev:
                    errors.append(f"{prefix}.best_evidence missing {field}")
        if isinstance(ev, dict):
            try:
                iou = float(ev.get("iou", 0.0))
                if not (0.0 <= iou <= 1.0):
                    errors.append(f"{prefix}.best_evidence.iou out of range")
            except (TypeError, ValueError):
                errors.append(f"{prefix}.best_evidence.iou must be numeric")
            try:
                score = float(ev.get("score_2d", 0.0))
                if not (0.0 <= score <= 1.0):
                    errors.append(f"{prefix}.best_evidence.score_2d out of range")
            except (TypeError, ValueError):
                errors.append(f"{prefix}.best_evidence.score_2d must be numeric")
    if errors:
        raise ValueError("invalid retrieval output:\n" + "\n".join(errors[:20]))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve relaxed 2D evidence for weak or unmatched 3D tracks")
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--detections-2d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relaxed-iou-threshold", type=float, default=0.15)
    parser.add_argument("--retrieval-min-score-2d", type=float, default=0.50)
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--debug-images", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/debug/retrieval_probe"))
    parser.add_argument("--max-debug-examples", type=int, default=30)
    parser.add_argument("--data-root", type=Path, default=None)
    return parser.parse_args()


def _print_summary(summary: Dict[str, Any], output_path: Path) -> None:
    print("Retrieval summary")
    print(f"- candidate tracks: {summary['candidate_tracks']}")
    print(f"- recovered tracks: {summary['recovered_tracks']}")
    print(f"- weak_temporal_evidence tracks: {summary['weak_temporal_evidence_tracks']}")
    print(f"- not_recovered tracks: {summary['not_recovered_tracks']}")
    print(f"- not_recovered_short_track: {summary['not_recovered_short_track']}")
    print(f"- not_recovered_no_2d_class_evidence: {summary['not_recovered_no_2d_class_evidence']}")
    print(f"- recovered by class: {summary['recovered_by_class']}")
    print(f"- average best IoU for recovered tracks: {summary['best_iou_avg']:.3f}")
    print(f"- average score_2d for recovered tracks: {summary['best_score_2d_avg']:.3f}")
    print(f"- box invalid/skipped count: {summary['box_invalid_count']}")
    print(f"- 2D boxes processed: {summary['detections_2d_processed']}")
    print(f"- 2D boxes clipped: {summary['detections_2d_clipped']}")
    print(f"- output: {output_path}")
    if summary.get("debug_image_dir"):
        print(f"- debug images: {summary['debug_image_dir']}")
    for warning in summary.get("warnings") or []:
        print(f"- warning: {warning}")


def main() -> int:
    args = parse_args()
    _doc, summary = run_retrieval(
        matching_path=args.matching,
        projection_path=args.projection,
        detections_2d_path=args.detections_2d,
        output_path=args.output,
        relaxed_iou_threshold=args.relaxed_iou_threshold,
        retrieval_min_score_2d=args.retrieval_min_score_2d,
        include_uncertain=args.include_uncertain,
        data_root=args.data_root,
        debug_images=args.debug_images,
        debug_dir=args.debug_dir,
        max_debug_examples=args.max_debug_examples,
    )
    _print_summary(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
