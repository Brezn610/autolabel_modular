from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from detector3d.detector_provider import load_detections_json


DEFAULT_THRESHOLDS = {
    "vehicle": 6.0,
    "pedestrian": 1.5,
}


@dataclass
class TrackObs:
    frame_id: str
    timestamp: int
    det_id: str
    box3d_lidar: List[float]
    score_3d: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": int(self.timestamp),
            "det_id": self.det_id,
            "box3d_lidar": [float(x) for x in self.box3d_lidar],
            "score_3d": float(self.score_3d),
        }


@dataclass
class ActiveTrack:
    track_id: int
    class_name: str
    observations: List[TrackObs] = field(default_factory=list)
    last_frame_index: int = 0
    missed_frames: int = 0

    @property
    def last_box(self) -> List[float]:
        return self.observations[-1].box3d_lidar

    def add(self, obs: TrackObs, frame_index: int) -> None:
        self.observations.append(obs)
        self.last_frame_index = int(frame_index)
        self.missed_frames = 0

    def to_output(self) -> Dict[str, Any]:
        scores = [o.score_3d for o in self.observations]
        return {
            "track_id": int(self.track_id),
            "class": self.class_name,
            "track_length": len(self.observations),
            "start_frame_id": self.observations[0].frame_id,
            "end_frame_id": self.observations[-1].frame_id,
            "start_timestamp": int(self.observations[0].timestamp),
            "end_timestamp": int(self.observations[-1].timestamp),
            "avg_score_3d": float(mean(scores)) if scores else 0.0,
            "frames": [obs.to_dict() for obs in self.observations],
        }


def _center_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def _obs_from_det(frame: Dict[str, Any], det: Dict[str, Any]) -> TrackObs:
    return TrackObs(
        frame_id=str(frame["frame_id"]),
        timestamp=int(frame["timestamp"]),
        det_id=str(det["det_id"]),
        box3d_lidar=[float(x) for x in det["box3d_lidar"]],
        score_3d=float(det["score_3d"]),
    )


def _match_detections_to_tracks(
    detections: List[Dict[str, Any]],
    active_tracks: List[ActiveTrack],
    thresholds: Dict[str, float],
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    candidates: List[Tuple[float, int, int]] = []
    for det_idx, det in enumerate(detections):
        cls = str(det["class"])
        threshold = float(thresholds.get(cls, 0.0))
        if threshold <= 0:
            continue
        box = det["box3d_lidar"]
        for track_idx, track in enumerate(active_tracks):
            if track.class_name != cls:
                continue
            dist = _center_distance(box, track.last_box)
            if dist <= threshold:
                candidates.append((dist, track_idx, det_idx))

    candidates.sort(key=lambda x: x[0])
    used_tracks = set()
    used_dets = set()
    matches: List[Tuple[int, int, float]] = []
    for dist, track_idx, det_idx in candidates:
        if track_idx in used_tracks or det_idx in used_dets:
            continue
        used_tracks.add(track_idx)
        used_dets.add(det_idx)
        matches.append((track_idx, det_idx, dist))

    unmatched_tracks = [i for i in range(len(active_tracks)) if i not in used_tracks]
    unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
    return matches, unmatched_tracks, unmatched_dets


def run_simple_tracker(
    detections_path: Path,
    *,
    vehicle_dist: float = DEFAULT_THRESHOLDS["vehicle"],
    pedestrian_dist: float = DEFAULT_THRESHOLDS["pedestrian"],
    max_age: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    det_doc = load_detections_json(detections_path)
    frames = sorted(det_doc["frames"], key=lambda f: (int(f["timestamp"]), str(f["frame_id"])))
    thresholds = {"vehicle": float(vehicle_dist), "pedestrian": float(pedestrian_dist)}

    active: List[ActiveTrack] = []
    finished: List[ActiveTrack] = []
    next_track_id = 0
    new_tracks_created = 0
    frame_assignment_rows: List[Dict[str, Any]] = []

    for frame_index, frame in enumerate(frames):
        detections = list(frame.get("boxes3d") or [])
        matches, unmatched_track_idxs, unmatched_det_idxs = _match_detections_to_tracks(
            detections,
            active,
            thresholds,
        )

        frame_objects: List[Dict[str, Any]] = []

        for track_idx, det_idx, _dist in matches:
            track = active[track_idx]
            det = detections[det_idx]
            obs = _obs_from_det(frame, det)
            track.add(obs, frame_index)
            frame_objects.append(_assignment_row(track.track_id, track.class_name, obs))

        for det_idx in unmatched_det_idxs:
            det = detections[det_idx]
            cls = str(det["class"])
            obs = _obs_from_det(frame, det)
            track = ActiveTrack(
                track_id=next_track_id,
                class_name=cls,
                observations=[obs],
                last_frame_index=frame_index,
                missed_frames=0,
            )
            next_track_id += 1
            new_tracks_created += 1
            active.append(track)
            frame_objects.append(_assignment_row(track.track_id, track.class_name, obs))

        unmatched_set = set(unmatched_track_idxs)
        still_active: List[ActiveTrack] = []
        for idx, track in enumerate(active):
            if idx in unmatched_set:
                track.missed_frames += 1
            if track.missed_frames > int(max_age):
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active

        frame_objects.sort(key=lambda x: (int(x["track_id"]), str(x["det_id"])))
        frame_assignment_rows.append(
            {
                "frame_id": str(frame["frame_id"]),
                "timestamp": int(frame["timestamp"]),
                "objects": frame_objects,
            }
        )

    finished.extend(active)
    finished.sort(key=lambda t: t.track_id)

    tracks_out = [track.to_output() for track in finished]
    summary = _summary(frames, tracks_out, new_tracks_created)
    track_doc = {
        "metadata": {
            "source": "simple_3d_tracker",
            "input": str(detections_path),
            "tracker": "class_aware_center_distance_greedy",
            "thresholds": thresholds,
            "max_age": int(max_age),
            "summary": summary,
        },
        "tracks": tracks_out,
    }
    frame_doc = {
        "metadata": {
            "source": "simple_3d_tracker",
            "input": str(detections_path),
            "tracker": "class_aware_center_distance_greedy",
            "thresholds": thresholds,
            "max_age": int(max_age),
        },
        "frames": frame_assignment_rows,
    }
    return track_doc, frame_doc, summary


def _assignment_row(track_id: int, class_name: str, obs: TrackObs) -> Dict[str, Any]:
    return {
        "track_id": int(track_id),
        "det_id": obs.det_id,
        "class": class_name,
        "box3d_lidar": [float(x) for x in obs.box3d_lidar],
        "score_3d": float(obs.score_3d),
    }


def _summary(frames: List[Dict[str, Any]], tracks: List[Dict[str, Any]], new_tracks_created: int) -> Dict[str, Any]:
    lengths = [int(t["track_length"]) for t in tracks]
    class_counts: Dict[str, int] = {}
    for t in tracks:
        class_counts[str(t["class"])] = class_counts.get(str(t["class"]), 0) + 1
    return {
        "input_frames": len(frames),
        "input_detections": sum(len(f.get("boxes3d") or []) for f in frames),
        "tracks": len(tracks),
        "vehicle_tracks": class_counts.get("vehicle", 0),
        "pedestrian_tracks": class_counts.get("pedestrian", 0),
        "average_track_length": float(mean(lengths)) if lengths else 0.0,
        "median_track_length": float(median(lengths)) if lengths else 0.0,
        "tracks_length_1": sum(1 for x in lengths if x == 1),
        "tracks_length_ge_3": sum(1 for x in lengths if x >= 3),
        "longest_track_length": max(lengths) if lengths else 0,
        "new_tracks_created": int(new_tracks_created),
    }


def _write_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple class-aware greedy 3D tracker")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-output", type=Path, required=True)
    parser.add_argument("--vehicle-dist", type=float, default=DEFAULT_THRESHOLDS["vehicle"])
    parser.add_argument("--pedestrian-dist", type=float, default=DEFAULT_THRESHOLDS["pedestrian"])
    parser.add_argument("--max-age", type=int, default=2)
    return parser.parse_args()


def _print_summary(summary: Dict[str, Any], track_doc: Dict[str, Any], output: Path, frame_output: Path) -> None:
    print("Simple 3D tracker summary")
    print(f"- input frames: {summary['input_frames']}")
    print(f"- input detections: {summary['input_detections']}")
    print(f"- tracks: {summary['tracks']}")
    print(f"- vehicle tracks: {summary['vehicle_tracks']}")
    print(f"- pedestrian tracks: {summary['pedestrian_tracks']}")
    print(f"- average track length: {summary['average_track_length']:.2f}")
    print(f"- median track length: {summary['median_track_length']:.2f}")
    print(f"- tracks length 1: {summary['tracks_length_1']}")
    print(f"- tracks length >= 3: {summary['tracks_length_ge_3']}")
    print(f"- longest track length: {summary['longest_track_length']}")
    print(f"- unmatched/new tracks created: {summary['new_tracks_created']}")
    print(f"Wrote tracks: {output}")
    print(f"Wrote frame assignments: {frame_output}")
    if track_doc["tracks"]:
        preview = dict(track_doc["tracks"][0])
        preview["frames"] = preview["frames"][:2]
        print("Preview track:")
        print(json.dumps(preview, ensure_ascii=False, indent=2)[:1600])


def main() -> int:
    args = parse_args()
    if not args.detections.is_file():
        raise FileNotFoundError(f"detections file not found: {args.detections}")
    track_doc, frame_doc, summary = run_simple_tracker(
        args.detections,
        vehicle_dist=args.vehicle_dist,
        pedestrian_dist=args.pedestrian_dist,
        max_age=args.max_age,
    )
    _write_json(args.output, track_doc)
    _write_json(args.frame_output, frame_doc)
    _print_summary(summary, track_doc, args.output, args.frame_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
