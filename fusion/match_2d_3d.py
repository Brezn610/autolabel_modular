from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2

from dataset.annotation_projection_probe import DEFAULT_CAMERAS, _camera_image_path, _load_timesync


VALID_CLASSES = {"vehicle", "pedestrian"}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def sanitize_box2d(box: Any) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    vals = [x1, y1, x2, y2]
    if not all(math.isfinite(v) for v in vals):
        return None
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def clip_box2d(
    box: Any,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Optional[List[float]]:
    clean = sanitize_box2d(box)
    if clean is None:
        return None
    x1, y1, x2, y2 = clean
    if image_width is not None and image_height is not None and image_width > 0 and image_height > 0:
        max_x = float(image_width - 1)
        max_y = float(image_height - 1)
        x1 = max(0.0, min(x1, max_x))
        x2 = max(0.0, min(x2, max_x))
        y1 = max(0.0, min(y1, max_y))
        y2 = max(0.0, min(y2, max_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_area(box: Any) -> float:
    clean = sanitize_box2d(box)
    if clean is None:
        return 0.0
    x1, y1, x2, y2 = clean
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_2d(
    box_a: Any,
    box_b: Any,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> float:
    a = clip_box2d(box_a, image_width=image_width, image_height=image_height)
    b = clip_box2d(box_b, image_width=image_width, image_height=image_height)
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    if union <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, inter / union)))


def _was_clipped(original: Any, clipped: Optional[List[float]]) -> bool:
    clean = sanitize_box2d(original)
    if clean is None or clipped is None:
        return False
    return any(abs(a - b) > 1e-6 for a, b in zip(clean, clipped))


def _class_compatible(track_class: Any, det_class: Any, source_label: Any = "") -> bool:
    track = str(track_class or "").strip().lower()
    det = str(det_class or "").strip().lower()
    label_tokens = set(str(source_label or "").strip().lower().replace(".", " ").split())
    if track == "vehicle":
        return det == "vehicle" or bool(label_tokens & {"vehicle", "car", "truck", "bus", "van", "suv"})
    if track == "pedestrian":
        return det == "pedestrian" or bool(label_tokens & {"pedestrian", "person"})
    return False


def _frames_by_id(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(frame.get("frame_id")): frame for frame in doc.get("frames") or []}


def _camera_names(projection_doc: Dict[str, Any], detections_doc: Dict[str, Any]) -> List[str]:
    names = list(projection_doc.get("metadata", {}).get("cameras") or [])
    if names:
        return names
    found: List[str] = []
    for frame in detections_doc.get("frames") or []:
        for camera in (frame.get("cameras") or {}).keys():
            if camera not in found:
                found.append(camera)
    return found or list(DEFAULT_CAMERAS)


def _infer_image_sizes(
    *,
    projection_doc: Dict[str, Any],
    data_root: Optional[Path],
    cameras: Iterable[str],
) -> Tuple[Dict[str, Tuple[int, int]], List[str], Dict[int, Dict[str, str]]]:
    root = data_root
    if root is None:
        meta_root = projection_doc.get("metadata", {}).get("data_root")
        root = Path(meta_root) if meta_root else None
    if root is None:
        return {}, ["image size unavailable: no --data-root and no projection metadata data_root"], {}
    try:
        timesync = _load_timesync(root)
    except Exception as exc:
        return {}, [f"image size unavailable: failed to load timesync from {root}: {exc}"], {}

    sizes: Dict[str, Tuple[int, int]] = {}
    warnings: List[str] = []
    timestamps = sorted(timesync)
    for camera in cameras:
        for ts in timestamps[:50]:
            image_name = (timesync.get(ts) or {}).get(camera)
            if not image_name:
                continue
            image_path = _camera_image_path(root, camera, image_name)
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            sizes[camera] = (int(w), int(h))
            break
        if camera not in sizes:
            warnings.append(f"image size unavailable for camera {camera}; IoU will not be boundary-clipped")
    return sizes, warnings, timesync


def _prepare_detections_for_frame(
    *,
    frame: Dict[str, Any],
    image_sizes: Dict[str, Tuple[int, int]],
    min_score_2d: float,
    summary: Counter[str],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for camera, detections in (frame.get("cameras") or {}).items():
        width, height = image_sizes.get(camera, (None, None))
        clean_dets: List[Dict[str, Any]] = []
        for det in detections or []:
            summary["detections_2d_processed"] += 1
            if float(det.get("score_2d", -1.0)) < min_score_2d:
                continue
            clipped = clip_box2d(det.get("box2d"), width, height)
            if clipped is None:
                summary["detections_2d_invalid"] += 1
                summary["malformed_boxes_skipped"] += 1
                continue
            if _was_clipped(det.get("box2d"), clipped):
                summary["detections_2d_clipped"] += 1
            clean_det = dict(det)
            clean_det["_box2d_clipped"] = clipped
            clean_det["_box_was_clipped"] = _was_clipped(det.get("box2d"), clipped)
            clean_dets.append(clean_det)
        out[camera] = clean_dets
    return out


def _status_for_track(visible_frames: int, ratio_frame: float) -> str:
    if visible_frames <= 0:
        return "not_visible"
    if ratio_frame >= 0.60:
        return "matched_high_confidence"
    if ratio_frame >= 0.30:
        return "matched_uncertain"
    return "unmatched"


def _aggregate_tracks(frame_matches: List[Dict[str, Any]], all_track_classes: Dict[int, str]) -> List[Dict[str, Any]]:
    by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in frame_matches:
        by_track[int(rec["track_id"])].append(rec)

    track_matches: List[Dict[str, Any]] = []
    for track_id in sorted(all_track_classes):
        rows = by_track.get(track_id, [])
        visible_pairs = len(rows)
        visible_frames_set = {str(r["frame_id"]) for r in rows}
        matched_rows = [r for r in rows if r.get("matched")]
        matched_frames_set = {str(r["frame_id"]) for r in matched_rows}
        ious = [float(r["iou"]) for r in matched_rows]
        scores = [float(r["score_2d"]) for r in matched_rows if r.get("score_2d") is not None]
        ratio_pair = len(matched_rows) / visible_pairs if visible_pairs else 0.0
        ratio_frame = len(matched_frames_set) / len(visible_frames_set) if visible_frames_set else 0.0
        best = max(matched_rows, key=lambda r: (float(r["iou"]), float(r.get("score_2d") or 0.0)), default=None)
        status = _status_for_track(len(visible_frames_set), ratio_frame)
        track_matches.append(
            {
                "track_id": track_id,
                "class": all_track_classes[track_id],
                "visible_frames": len(visible_frames_set),
                "matched_frames": len(matched_frames_set),
                "match_ratio_frame": float(ratio_frame),
                "visible_frame_camera_pairs": visible_pairs,
                "matched_frame_camera_pairs": len(matched_rows),
                "match_ratio_frame_camera": float(ratio_pair),
                "avg_iou": float(mean(ious)) if ious else 0.0,
                "max_iou": float(max(ious)) if ious else 0.0,
                "avg_score_2d": float(mean(scores)) if scores else 0.0,
                "status": status,
                "best_evidence": (
                    {
                        "frame_id": str(best["frame_id"]),
                        "camera": str(best["camera"]),
                        "matched_2d_id": str(best["matched_2d_id"]),
                        "iou": float(best["iou"]),
                        "score_2d": float(best["score_2d"]),
                        "source_label": str(best.get("source_label", "")),
                    }
                    if best
                    else None
                ),
            }
        )
    return track_matches


def match_2d_3d(
    *,
    projection_path: Path,
    detections_2d_path: Path,
    output_path: Path,
    iou_threshold: float = 0.30,
    min_score_2d: float = 0.30,
    data_root: Optional[Path] = None,
    debug_images: bool = False,
    debug_dir: Path = Path("outputs/debug/2d_3d_matching_probe"),
    max_debug_frames: int = 10,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    projection_doc = _read_json(projection_path)
    detections_doc = _read_json(detections_2d_path)
    det_frames = _frames_by_id(detections_doc)
    cameras = _camera_names(projection_doc, detections_doc)
    image_sizes, warnings, timesync = _infer_image_sizes(
        projection_doc=projection_doc,
        data_root=data_root,
        cameras=cameras,
    )

    summary_counts: Counter[str] = Counter()
    frame_matches: List[Dict[str, Any]] = []
    all_track_classes: Dict[int, str] = {}
    debug_frame_ids: List[str] = []

    for frame_idx, proj_frame in enumerate(projection_doc.get("frames") or []):
        frame_id = str(proj_frame.get("frame_id"))
        timestamp = int(proj_frame.get("timestamp", 0))
        det_frame = det_frames.get(frame_id, {"cameras": {}})
        detections_by_camera = _prepare_detections_for_frame(
            frame=det_frame,
            image_sizes=image_sizes,
            min_score_2d=min_score_2d,
            summary=summary_counts,
        )
        if debug_images and len(debug_frame_ids) < max_debug_frames:
            debug_frame_ids.append(frame_id)

        for obj in proj_frame.get("objects") or []:
            track_id = int(obj["track_id"])
            cls = str(obj.get("class", ""))
            all_track_classes.setdefault(track_id, cls)
            projections = obj.get("camera_projections") or {}
            for camera in cameras:
                proj = projections.get(camera) or {}
                if not proj.get("in_fov"):
                    continue
                summary_counts["projected_boxes_processed"] += 1
                width, height = image_sizes.get(camera, (None, None))
                projected_box = clip_box2d(proj.get("projected_box2d"), width, height)
                if projected_box is None:
                    summary_counts["projected_boxes_invalid"] += 1
                    summary_counts["malformed_boxes_skipped"] += 1
                    continue
                proj_clipped = _was_clipped(proj.get("projected_box2d"), projected_box)
                if proj_clipped:
                    summary_counts["projected_boxes_clipped"] += 1

                best_det: Optional[Dict[str, Any]] = None
                best_iou = 0.0
                for det in detections_by_camera.get(camera) or []:
                    if not _class_compatible(cls, det.get("class"), det.get("source_label", "")):
                        continue
                    iou = iou_2d(projected_box, det.get("_box2d_clipped"), width, height)
                    if iou > best_iou:
                        best_iou = iou
                        best_det = det

                matched = best_det is not None and best_iou >= iou_threshold
                rec = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "track_id": track_id,
                    "det_id": str(obj.get("det_id", "")),
                    "class": cls,
                    "camera": camera,
                    "projected_box2d": projected_box,
                    "matched": bool(matched),
                    "matched_2d_id": str(best_det["det2d_id"]) if matched and best_det else None,
                    "iou": float(best_iou if matched else 0.0),
                    "score_2d": float(best_det["score_2d"]) if matched and best_det else None,
                    "source_label": str(best_det.get("source_label", "")) if matched and best_det else "",
                    "box_was_clipped": bool(proj_clipped or (bool(best_det.get("_box_was_clipped")) if matched and best_det else False)),
                }
                frame_matches.append(rec)

    track_matches = _aggregate_tracks(frame_matches, all_track_classes)
    status_counts = Counter(t["status"] for t in track_matches)
    class_counts = Counter(t["class"] for t in track_matches)
    class_stats = _per_class_stats(track_matches)
    ratios = [float(t["match_ratio_frame"]) for t in track_matches if t["status"] != "not_visible"]

    summary = {
        "num_tracks": len(track_matches),
        "vehicle_tracks": int(class_counts["vehicle"]),
        "pedestrian_tracks": int(class_counts["pedestrian"]),
        "matched_high_confidence": int(status_counts["matched_high_confidence"]),
        "matched_uncertain": int(status_counts["matched_uncertain"]),
        "unmatched": int(status_counts["unmatched"]),
        "not_visible": int(status_counts["not_visible"]),
        "total_visible_frame_camera_pairs": int(sum(t["visible_frame_camera_pairs"] for t in track_matches)),
        "total_matched_frame_camera_pairs": int(sum(t["matched_frame_camera_pairs"] for t in track_matches)),
        "average_match_ratio_frame": float(mean(ratios)) if ratios else 0.0,
        "per_class_match_stats": class_stats,
        "projected_boxes_processed": int(summary_counts["projected_boxes_processed"]),
        "projected_boxes_clipped": int(summary_counts["projected_boxes_clipped"]),
        "projected_boxes_invalid": int(summary_counts["projected_boxes_invalid"]),
        "detections_2d_processed": int(summary_counts["detections_2d_processed"]),
        "detections_2d_clipped": int(summary_counts["detections_2d_clipped"]),
        "detections_2d_invalid": int(summary_counts["detections_2d_invalid"]),
        "malformed_boxes_skipped": int(summary_counts["malformed_boxes_skipped"]),
        "warnings": warnings,
        "debug_image_dir": str(debug_dir) if debug_images else "",
    }

    out_doc = {
        "metadata": {
            "source": "match_2d_3d",
            "projection_input": str(projection_path),
            "detections_2d_input": str(detections_2d_path),
            "normal_iou_threshold": float(iou_threshold),
            "min_score_2d": float(min_score_2d),
            "note": "Track-level 2D-3D matching using projected 3D boxes and GroundingDINO 2D boxes",
            "warnings": warnings,
        },
        "summary": summary,
        "track_matches": track_matches,
        "frame_matches": frame_matches,
    }
    validate_matching_doc(out_doc)
    _write_json(output_path, out_doc)

    if debug_images:
        _write_debug_images(
            projection_doc=projection_doc,
            detections_doc=detections_doc,
            frame_matches=frame_matches,
            debug_frame_ids=debug_frame_ids,
            data_root=data_root or Path(projection_doc.get("metadata", {}).get("data_root", "")),
            timesync=timesync,
            debug_dir=debug_dir,
            image_sizes=image_sizes,
        )
    return out_doc, summary


def _per_class_stats(track_matches: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in track_matches:
        by_class[str(item["class"])].append(item)
    for cls, rows in sorted(by_class.items()):
        statuses = Counter(row["status"] for row in rows)
        visible_rows = [row for row in rows if row["status"] != "not_visible"]
        stats[cls] = {
            "num_tracks": len(rows),
            "matched_high_confidence": int(statuses["matched_high_confidence"]),
            "matched_uncertain": int(statuses["matched_uncertain"]),
            "unmatched": int(statuses["unmatched"]),
            "not_visible": int(statuses["not_visible"]),
            "avg_match_ratio_frame": float(mean(float(r["match_ratio_frame"]) for r in visible_rows)) if visible_rows else 0.0,
        }
    return stats


def _write_debug_images(
    *,
    projection_doc: Dict[str, Any],
    detections_doc: Dict[str, Any],
    frame_matches: List[Dict[str, Any]],
    debug_frame_ids: List[str],
    data_root: Path,
    timesync: Dict[int, Dict[str, str]],
    debug_dir: Path,
    image_sizes: Dict[str, Tuple[int, int]],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    det_frames = _frames_by_id(detections_doc)
    matches_by_frame_camera: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in frame_matches:
        if str(rec["frame_id"]) in debug_frame_ids:
            matches_by_frame_camera[(str(rec["frame_id"]), str(rec["camera"]))].append(rec)

    for frame in projection_doc.get("frames") or []:
        frame_id = str(frame.get("frame_id"))
        if frame_id not in debug_frame_ids:
            continue
        timestamp = int(frame.get("timestamp", 0))
        files = timesync.get(timestamp) or {}
        det_frame = det_frames.get(frame_id, {"cameras": {}})
        cameras = set((det_frame.get("cameras") or {}).keys())
        cameras.update((projection_doc.get("metadata", {}).get("cameras") or []))
        for camera in sorted(cameras):
            image_name = files.get(camera)
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
            for det in (det_frame.get("cameras") or {}).get(camera, []) or []:
                det_box = clip_box2d(det.get("box2d"), image.shape[1], image.shape[0])
                if det_box is None:
                    continue
                _draw_rect(canvas, det_box, (255, 160, 0), f"2D {det.get('class')} {float(det.get('score_2d', 0.0)):.2f}", 1)
            for rec in matches_by_frame_camera.get((frame_id, camera), []):
                color = (0, 255, 0) if rec.get("matched") else (0, 0, 255)
                label = (
                    f"t{rec['track_id']} {rec['class']} IoU {float(rec['iou']):.2f} "
                    f"s {float(rec['score_2d'] or 0.0):.2f}"
                    if rec.get("matched")
                    else f"t{rec['track_id']} {rec['class']} unmatched"
                )
                _draw_rect(canvas, rec["projected_box2d"], color, label, 2)
            cv2.imwrite(str(debug_dir / f"frame_{frame_id}__{camera}.jpg"), canvas)


def _draw_rect(image: Any, box: List[float], color: Tuple[int, int, int], label: str, thickness: int) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.putText(image, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def validate_matching_doc(doc: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(doc.get("metadata"), dict):
        errors.append("metadata must exist")
    if not isinstance(doc.get("track_matches"), list):
        errors.append("track_matches must exist and be a list")
    if not isinstance(doc.get("frame_matches"), list):
        errors.append("frame_matches must exist and be a list")
    for idx, item in enumerate(doc.get("track_matches") or []):
        for field in ("track_id", "class", "status"):
            if field not in item:
                errors.append(f"track_matches[{idx}] missing {field}")
    for idx, item in enumerate(doc.get("frame_matches") or []):
        for field in ("frame_id", "track_id", "camera", "matched"):
            if field not in item:
                errors.append(f"frame_matches[{idx}] missing {field}")
        try:
            iou = float(item.get("iou", 0.0))
            if not (0.0 <= iou <= 1.0):
                errors.append(f"frame_matches[{idx}].iou out of range")
        except (TypeError, ValueError):
            errors.append(f"frame_matches[{idx}].iou must be numeric")
        if item.get("matched"):
            for field in ("matched_2d_id", "iou", "score_2d"):
                if item.get(field) is None:
                    errors.append(f"frame_matches[{idx}] matched=true missing {field}")
            try:
                score = float(item.get("score_2d"))
                if not (0.0 <= score <= 1.0):
                    errors.append(f"frame_matches[{idx}].score_2d out of range")
            except (TypeError, ValueError):
                errors.append(f"frame_matches[{idx}].score_2d must be numeric")
    if errors:
        raise ValueError("invalid matching output:\n" + "\n".join(errors[:20]))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match projected 3D track boxes to canonical 2D detections")
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--detections-2d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--min-score-2d", type=float, default=0.30)
    parser.add_argument("--debug-images", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/debug/2d_3d_matching_probe"))
    parser.add_argument("--max-debug-frames", type=int, default=10)
    parser.add_argument("--data-root", type=Path, default=None)
    return parser.parse_args()


def _print_summary(summary: Dict[str, Any], output_path: Path) -> None:
    print("2D-3D matching summary")
    print(f"- tracks: {summary['num_tracks']}")
    print(f"- vehicle tracks: {summary['vehicle_tracks']}")
    print(f"- pedestrian tracks: {summary['pedestrian_tracks']}")
    print(f"- total visible frame-camera pairs: {summary['total_visible_frame_camera_pairs']}")
    print(f"- total matched frame-camera pairs: {summary['total_matched_frame_camera_pairs']}")
    print(f"- matched_high_confidence tracks: {summary['matched_high_confidence']}")
    print(f"- matched_uncertain tracks: {summary['matched_uncertain']}")
    print(f"- unmatched tracks: {summary['unmatched']}")
    print(f"- not_visible tracks: {summary['not_visible']}")
    print(f"- average match_ratio_frame: {summary['average_match_ratio_frame']:.3f}")
    print(f"- per-class match stats: {summary['per_class_match_stats']}")
    print(f"- projected boxes processed: {summary['projected_boxes_processed']}")
    print(f"- projected boxes clipped: {summary['projected_boxes_clipped']}")
    print(f"- projected boxes invalid/skipped: {summary['projected_boxes_invalid']}")
    print(f"- 2D boxes processed: {summary['detections_2d_processed']}")
    print(f"- 2D boxes clipped: {summary['detections_2d_clipped']}")
    print(f"- 2D boxes invalid/skipped: {summary['detections_2d_invalid']}")
    print(f"- malformed boxes skipped: {summary['malformed_boxes_skipped']}")
    print(f"- output: {output_path}")
    if summary.get("debug_image_dir"):
        print(f"- debug images: {summary['debug_image_dir']}")
    for warning in summary.get("warnings") or []:
        print(f"- warning: {warning}")


def main() -> int:
    args = parse_args()
    _doc, summary = match_2d_3d(
        projection_path=args.projection,
        detections_2d_path=args.detections_2d,
        output_path=args.output,
        iou_threshold=args.iou_threshold,
        min_score_2d=args.min_score_2d,
        data_root=args.data_root,
        debug_images=args.debug_images,
        debug_dir=args.debug_dir,
        max_debug_frames=args.max_debug_frames,
    )
    _print_summary(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
