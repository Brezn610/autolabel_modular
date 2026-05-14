from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from dataset.annotation_bev_probe import _world_to_px
from dataset.annotation_projection_probe import _camera_image_path, _corners_from_center_lwh_yaw, _load_timesync


COLORS = {
    "projected": (255, 80, 40),
    "dino": (80, 220, 80),
    "matched": (120, 255, 255),
    "recovered": (0, 165, 255),
    "rejected": (0, 0, 255),
    "bev_box": (255, 80, 40),
    "text": (235, 235, 235),
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_lidar_xyz(data_root: Path, lidar_filename: str) -> np.ndarray:
    path = data_root / "middle_lidar" / lidar_filename
    if not path.is_file():
        raise FileNotFoundError(f"LiDAR file not found: {path}")
    npz = np.load(path)
    if not all(k in npz.files for k in ("x", "y", "z")):
        raise ValueError(f"LiDAR npz must contain x,y,z arrays: {path}")
    return np.stack([npz["x"], npz["y"], npz["z"]], axis=1).astype(np.float32)


def _downsample_points(xyz: np.ndarray, max_points: int) -> np.ndarray:
    if xyz.shape[0] <= max_points:
        return xyz
    rng = np.random.default_rng(0)
    idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[idx]


def _write_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def _frames_by_id(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(frame.get("frame_id")): frame for frame in doc.get("frames") or []}


def _track_rows_by_id(doc: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(row["track_id"]): row for row in doc.get("tracks") or []}


def _rows_by_track_id(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(row["track_id"]): row for row in rows}


def _select_track(tracks: Dict[int, Dict[str, Any]], status: str, preferred: int) -> Optional[Dict[str, Any]]:
    row = tracks.get(preferred)
    if row and row.get("final_status") == status and row.get("class") == "vehicle":
        return row
    for candidate in tracks.values():
        if candidate.get("final_status") == status and candidate.get("class") == "vehicle":
            return candidate
    for candidate in tracks.values():
        if candidate.get("final_status") == status:
            return candidate
    return None


def _representative_frame_ids(track_id: int, frame_assignments: Dict[str, Dict[str, Any]], count: int = 4) -> List[str]:
    frame_ids = []
    for frame_id, frame in sorted(frame_assignments.items()):
        if any(int(obj["track_id"]) == track_id for obj in frame.get("objects") or []):
            frame_ids.append(frame_id)
    if len(frame_ids) <= count:
        return frame_ids
    idxs = np.linspace(0, len(frame_ids) - 1, count).round().astype(int)
    out: List[str] = []
    for idx in idxs:
        fid = frame_ids[int(idx)]
        if fid not in out:
            out.append(fid)
    return out


def _find_track_object(frame: Dict[str, Any], track_id: int) -> Optional[Dict[str, Any]]:
    for obj in frame.get("objects") or []:
        if int(obj.get("track_id", -1)) == track_id:
            return obj
    return None


def _best_camera_for_frame(
    *,
    frame_id: str,
    track_id: int,
    projection_frame: Dict[str, Any],
    track_summary: Dict[str, Any],
    matching_rows: List[Dict[str, Any]],
    retrieval_row: Optional[Dict[str, Any]],
) -> Optional[str]:
    ev = track_summary.get("retrieval_best_evidence")
    if isinstance(ev, dict) and ev.get("frame_id") == frame_id:
        return str(ev.get("camera"))
    if retrieval_row and isinstance(retrieval_row.get("best_evidence"), dict):
        rev = retrieval_row["best_evidence"]
        if rev.get("frame_id") == frame_id:
            return str(rev.get("camera"))
    matched = [r for r in matching_rows if str(r.get("frame_id")) == frame_id and int(r.get("track_id", -1)) == track_id and r.get("matched")]
    if matched:
        return str(max(matched, key=lambda r: float(r.get("iou", 0.0))).get("camera"))
    obj = _find_track_object(projection_frame, track_id)
    if obj:
        for camera, proj in (obj.get("camera_projections") or {}).items():
            if proj.get("in_fov") and proj.get("projected_box2d"):
                return str(camera)
    return None


def _draw_rect(img: np.ndarray, box: Any, color: Tuple[int, int, int], label: str, thickness: int = 2) -> None:
    if not isinstance(box, list) or len(box) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    h, w = img.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.putText(img, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def _render_camera_panel(
    *,
    data_root: Path,
    timesync: Dict[int, Dict[str, str]],
    frame_id: str,
    timestamp: int,
    camera: str,
    track_id: int,
    final_status: str,
    projection_frame: Dict[str, Any],
    detections_frame: Dict[str, Any],
    matching_rows: List[Dict[str, Any]],
    retrieval_row: Optional[Dict[str, Any]],
    size: Tuple[int, int] = (640, 360),
) -> Tuple[np.ndarray, Optional[str]]:
    image_name = (timesync.get(timestamp) or {}).get(camera)
    if not image_name:
        return _blank_panel(size, f"missing image\n{frame_id} {camera}"), "missing_image"
    image_path = _camera_image_path(data_root, camera, image_name)
    img = cv2.imread(str(image_path))
    if img is None:
        return _blank_panel(size, f"failed image\n{image_path.name}"), "failed_image"
    obj = _find_track_object(projection_frame, track_id)
    for det in (detections_frame.get("cameras") or {}).get(camera, []) or []:
        _draw_rect(img, det.get("box2d"), COLORS["dino"], f"2D {det.get('class')} {float(det.get('score_2d', 0.0)):.2f}", 1)
    if obj:
        proj = (obj.get("camera_projections") or {}).get(camera) or {}
        if proj.get("in_fov"):
            color = COLORS["projected"]
            if final_status == "recovered":
                color = COLORS["recovered"]
            elif final_status == "rejected":
                color = COLORS["rejected"]
            _draw_rect(img, proj.get("projected_box2d"), color, f"t{track_id} {final_status}", 2)
    matches = [r for r in matching_rows if str(r.get("frame_id")) == frame_id and int(r.get("track_id", -1)) == track_id and r.get("camera") == camera]
    best = max(matches, key=lambda r: float(r.get("iou", 0.0)), default=None)
    if best and best.get("matched"):
        _draw_rect(img, best.get("projected_box2d"), COLORS["matched"], f"match IoU {float(best.get('iou', 0.0)):.2f}", 3)
    if retrieval_row and isinstance(retrieval_row.get("best_evidence"), dict):
        ev = retrieval_row["best_evidence"]
        if ev.get("frame_id") == frame_id and ev.get("camera") == camera:
            _draw_rect(img, ev.get("projected_box2d"), COLORS["recovered"], f"retrieval IoU {float(ev.get('iou', 0.0)):.2f}", 3)
    cv2.putText(img, f"frame {frame_id} | {camera}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS["text"], 2, cv2.LINE_AA)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), None


def _blank_panel(size: Tuple[int, int], text: str) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), 28, dtype=np.uint8)
    y = 40
    for line in text.splitlines():
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text"], 2, cv2.LINE_AA)
        y += 34
    return img


def _render_bev_panel(
    *,
    data_root: Path,
    timesync: Dict[int, Dict[str, str]],
    frame: Dict[str, Any],
    track_id: int,
    final_status: str,
    size: Tuple[int, int] = (640, 360),
) -> Tuple[np.ndarray, Optional[str]]:
    timestamp = int(frame.get("timestamp", 0))
    lidar_name = (timesync.get(timestamp) or {}).get("middle_lidar")
    if not lidar_name:
        lidar_name = f"{timestamp}.npz"
    try:
        points = _load_lidar_xyz(data_root, lidar_name)
    except Exception as exc:  # noqa: BLE001
        return _blank_panel(size, f"missing lidar\n{type(exc).__name__}: {exc}"), "missing_lidar"
    points = _downsample_points(points, 25000)
    obj = _find_track_object(frame, track_id)
    if not obj:
        return _blank_panel(size, f"missing track in frame\ntrack {track_id}"), "missing_track_bev"
    box = [float(v) for v in obj["box3d_lidar"]]
    corners = _corners_from_center_lwh_yaw(box[:3], box[3:6], box[6])[:4, :2]
    xy = points[:, :2]
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    all_xy = np.vstack([xy, corners])
    pad = 6.0
    x_range = (float(np.percentile(all_xy[:, 0], 1)) - pad, float(np.percentile(all_xy[:, 0], 99)) + pad)
    y_range = (float(np.percentile(all_xy[:, 1], 1)) - pad, float(np.percentile(all_xy[:, 1], 99)) + pad)
    w, h = size
    canvas = np.full((h, w, 3), 24, dtype=np.uint8)
    pts_px = _world_to_px(xy, x_range, y_range, min(w, h))
    xoff = (w - min(w, h)) // 2
    for p in pts_px.astype(np.int32):
        u, v = int(p[0]) + xoff, int(p[1])
        if 0 <= u < w and 0 <= v < h:
            canvas[v, u] = (95, 95, 95)
    rect_px = _world_to_px(corners, x_range, y_range, min(w, h)).astype(np.int32)
    rect_px[:, 0] += xoff
    color = COLORS["bev_box"] if final_status != "rejected" else COLORS["rejected"]
    cv2.polylines(canvas, [rect_px.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"BEV t{track_id} {final_status}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS["text"], 2, cv2.LINE_AA)
    return canvas, None


def _info_panel(track: Dict[str, Any], retrieval_row: Optional[Dict[str, Any]], size: Tuple[int, int] = (640, 360)) -> np.ndarray:
    status = str(track["final_status"])
    explanation = {
        "accepted": "strong 2D-3D matching",
        "recovered": "strict matching failed, retrieval recovered the track",
        "rejected": "insufficient 2D evidence / short or unsupported track",
        "uncertain": "partial evidence; kept for review",
    }.get(status, "")
    lines = [
        f"track_id: {track['track_id']}",
        f"class: {track['class']}",
        f"track_length: {track['track_length']}",
        f"final_status: {track['final_status']}",
        f"final_score: {float(track['final_score']):.3f}",
        f"matching_status: {track.get('matching_status')}",
        f"retrieval_status: {track.get('retrieval_status')}",
        f"avg_score_3d: {float(track.get('avg_score_3d', 0.0)):.3f}",
        f"match_ratio_frame: {float(track.get('match_ratio_frame', 0.0)):.3f}",
        f"avg_iou: {float(track.get('avg_iou', 0.0)):.3f}",
    ]
    ev = track.get("retrieval_best_evidence") or (retrieval_row or {}).get("best_evidence")
    if isinstance(ev, dict):
        lines.append(f"best retrieval IoU: {float(ev.get('iou', 0.0)):.3f}")
    lines.append("")
    lines.append(explanation)
    panel = np.full((size[1], size[0], 3), 35, dtype=np.uint8)
    y = 32
    for line in lines:
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLORS["text"], 1, cv2.LINE_AA)
        y += 28
    return panel


def render_case_studies(
    *,
    data_root: Path,
    frame_assignments_path: Path,
    projection_path: Path,
    detections_2d_path: Path,
    matching_path: Path,
    retrieval_path: Path,
    fusion_tracks_path: Path,
    output_dir: Path,
    manifest_path: Path = Path("outputs/visualization/visualization_manifest.json"),
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timesync = _load_timesync(data_root)
    frames_assign = _frames_by_id(_read_json(frame_assignments_path))
    projection_frames = _frames_by_id(_read_json(projection_path))
    detections_frames = _frames_by_id(_read_json(detections_2d_path))
    matching_doc = _read_json(matching_path)
    retrieval_doc = _read_json(retrieval_path)
    tracks = _track_rows_by_id(_read_json(fusion_tracks_path))
    retrieval_by_track = _rows_by_track_id(retrieval_doc.get("track_retrieval") or [])
    matching_rows = matching_doc.get("frame_matches") or []

    selected = {
        "accepted": _select_track(tracks, "accepted", 10),
        "recovered": _select_track(tracks, "recovered", 168),
        "rejected": _select_track(tracks, "rejected", 1),
    }
    generated: List[str] = []
    missing: List[Dict[str, Any]] = []

    for status, track in selected.items():
        if not track:
            missing.append({"status": status, "reason": "no_track_found"})
            continue
        track_id = int(track["track_id"])
        frame_ids = _representative_frame_ids(track_id, frames_assign, count=4)
        camera_panels: List[np.ndarray] = []
        for frame_id in frame_ids[:4]:
            proj_frame = projection_frames.get(frame_id, {})
            camera = _best_camera_for_frame(
                frame_id=frame_id,
                track_id=track_id,
                projection_frame=proj_frame,
                track_summary=track,
                matching_rows=matching_rows,
                retrieval_row=retrieval_by_track.get(track_id),
            )
            if not camera:
                missing.append({"track_id": track_id, "frame_id": frame_id, "reason": "no_in_fov_camera"})
                continue
            panel, err = _render_camera_panel(
                data_root=data_root,
                timesync=timesync,
                frame_id=frame_id,
                timestamp=int(proj_frame.get("timestamp", frames_assign.get(frame_id, {}).get("timestamp", 0))),
                camera=camera,
                track_id=track_id,
                final_status=status,
                projection_frame=proj_frame,
                detections_frame=detections_frames.get(frame_id, {}),
                matching_rows=matching_rows,
                retrieval_row=retrieval_by_track.get(track_id),
            )
            if err:
                missing.append({"track_id": track_id, "frame_id": frame_id, "camera": camera, "reason": err})
            camera_panels.append(panel)
        while len(camera_panels) < 4:
            camera_panels.append(_blank_panel((640, 360), "no panel"))
        bev_frame = frames_assign.get(frame_ids[len(frame_ids) // 2] if frame_ids else "", {})
        bev_panel, err = _render_bev_panel(
            data_root=data_root,
            timesync=timesync,
            frame=bev_frame,
            track_id=track_id,
            final_status=status,
        )
        if err:
            missing.append({"track_id": track_id, "reason": err})
        info = _info_panel(track, retrieval_by_track.get(track_id))
        top = np.hstack(camera_panels[:2])
        mid = np.hstack(camera_panels[2:4])
        bottom = np.hstack([bev_panel, info])
        figure = np.vstack([top, mid, bottom])
        out_path = output_dir / f"{status}_track_{track_id}.png"
        cv2.imwrite(str(out_path), figure)
        generated.append(str(out_path))

    manifest = {
        "selected_tracks": {
            status: int(track["track_id"]) if track else None for status, track in selected.items()
        },
        "generated_files": generated,
        "missing_or_failed_items": missing,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render static case-study panels for fused tracks")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=False)
    parser.add_argument("--frame-assignments", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--detections-2d", type=Path, required=True)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--fusion-tracks", type=Path, required=True)
    parser.add_argument("--fusion-labels", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/visualization/visualization_manifest.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = render_case_studies(
        data_root=args.data_root,
        frame_assignments_path=args.frame_assignments,
        projection_path=args.projection,
        detections_2d_path=args.detections_2d,
        matching_path=args.matching,
        retrieval_path=args.retrieval,
        fusion_tracks_path=args.fusion_tracks,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print("Case-study visualization summary")
    print(f"- selected tracks: {manifest['selected_tracks']}")
    print(f"- generated files: {len(manifest['generated_files'])}")
    for path in manifest["generated_files"]:
        print(f"  {path}")
    print(f"- missing/failed items: {len(manifest['missing_or_failed_items'])}")
    print(f"- manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
