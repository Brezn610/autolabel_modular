# Detector-Driven Auto-Labeling Mainline

This branch is the detector-driven multi-modal auto-labeling prototype.

The previous mainline:

```text
GroundingDINO 2D detection
-> LiDAR frustum point selection
-> DBSCAN
-> OBB fitting
-> old pseudo labels
```

has been removed from the active code path. The new mainline does not use
DBSCAN, OBB fitting, or frustum lifting as the 3D source.

## Current Design

The current 3D branch consumes detector-like 3D boxes in a canonical JSON
schema:

```text
[x, y, z, length, width, height, yaw]
```

For this prototype, the 3D detector provider is:

```text
detector3d.mock_lidar_detector
```

It generates detector-like 3D outputs from `annotations.json` with missed
detections, localization noise, yaw noise, size noise, false positives, and
confidence scores. Coordinates are treated as `lidar_like` based on the BEV
diagnostics performed during this refactor; the dataset itself does not
explicitly declare the annotation coordinate frame.

Future real detector adapters such as CenterPoint, TransFusion, or MS3D should
convert their outputs into the same canonical schema. Downstream tracking,
projection, matching, retrieval, fusion, evaluation, and visualization should
not care whether boxes came from the mock provider or a real detector.

## Pipeline Commands

Generate mock 3D detector output:

```bash
python -m detector3d.mock_lidar_detector \
  --annotations annotations.json \
  --output outputs/detections_3d/mock_lidar_detector_output_100f.json \
  --seed 42 \
  --max-frames 100
```

Track 3D detections:

```bash
python -m tracking.simple_3d_tracker \
  --detections outputs/detections_3d/mock_lidar_detector_output_100f.json \
  --output outputs/tracks/tracks_3d_100f_tuned.json \
  --frame-output outputs/tracks/frame_track_assignments_100f_tuned.json \
  --vehicle-dist 6.0 \
  --pedestrian-dist 1.5 \
  --max-age 2
```

Project 3D tracks into cameras:

```bash
python -m fusion.project_tracks \
  --tracks outputs/tracks/frame_track_assignments_100f_tuned.json \
  --calibration calibration.json \
  --data-root /mnt/d/thesis/day \
  --output outputs/projection/track_projection_100f.json
```

Run GroundingDINO 2D detections:

```bash
python -m detect.run_groundingdino_2d \
  --data-root /mnt/d/thesis/day \
  --output outputs/detections_2d/groundingdino_2d_100f.json \
  --max-frames 100
```

Match projected 3D tracks to 2D detections:

```bash
python -m fusion.match_2d_3d \
  --projection outputs/projection/track_projection_100f.json \
  --detections-2d outputs/detections_2d/groundingdino_2d_100f.json \
  --output outputs/matching/track_level_matching_100f.json \
  --iou-threshold 0.30 \
  --min-score-2d 0.30 \
  --data-root /mnt/d/thesis/day
```

Retrieve relaxed evidence for unmatched tracks:

```bash
python -m fusion.retrieval \
  --matching outputs/matching/track_level_matching_100f.json \
  --projection outputs/projection/track_projection_100f.json \
  --detections-2d outputs/detections_2d/groundingdino_2d_100f.json \
  --output outputs/retrieval/retrieval_results_100f.json \
  --relaxed-iou-threshold 0.15 \
  --retrieval-min-score-2d 0.50 \
  --data-root /mnt/d/thesis/day
```

Fuse confidence into pseudo labels:

```bash
python -m fusion.confidence_fusion \
  --tracks outputs/tracks/tracks_3d_100f_tuned.json \
  --frame-assignments outputs/tracks/frame_track_assignments_100f_tuned.json \
  --matching outputs/matching/track_level_matching_100f.json \
  --retrieval outputs/retrieval/retrieval_results_100f.json \
  --output-labels outputs/fusion/fused_pseudo_labels_100f.json \
  --output-tracks outputs/fusion/fused_track_summary_100f.json
```

Evaluate pseudo labels:

```bash
python -m evaluation.eval_pseudo_labels \
  --predictions outputs/fusion/fused_pseudo_labels_100f.json \
  --gt annotations.json \
  --output-json outputs/evaluation/evaluation_summary_100f.json \
  --output-csv outputs/evaluation/ablation_table_100f.csv \
  --vehicle-threshold 2.0 \
  --pedestrian-threshold 1.0
```

Render static report/demo visualizations:

```bash
python -m visualization.visualize_case_studies \
  --data-root /mnt/d/thesis/day \
  --calibration calibration.json \
  --frame-assignments outputs/tracks/frame_track_assignments_100f_tuned.json \
  --projection outputs/projection/track_projection_100f.json \
  --detections-2d outputs/detections_2d/groundingdino_2d_100f.json \
  --matching outputs/matching/track_level_matching_100f.json \
  --retrieval outputs/retrieval/retrieval_results_100f.json \
  --fusion-tracks outputs/fusion/fused_track_summary_100f.json \
  --fusion-labels outputs/fusion/fused_pseudo_labels_100f.json \
  --output-dir outputs/visualization/case_studies
```

```bash
python -m visualization.visualize_summary_panel \
  --fusion-tracks outputs/fusion/fused_track_summary_100f.json \
  --evaluation outputs/evaluation/evaluation_summary_100f.json \
  --output-dir outputs/visualization/summary_panel
```

## Current Outputs

The current 100-frame demo artifacts live under `outputs/`:

- `outputs/detections_3d/`
- `outputs/tracks/`
- `outputs/projection/`
- `outputs/detections_2d/`
- `outputs/matching/`
- `outputs/retrieval/`
- `outputs/fusion/`
- `outputs/evaluation/`
- `outputs/visualization/`

Old `out/` and `results/` artifacts from the DBSCAN/OBB/frustum pipeline were
removed from the mainline cleanup.

## Notes And Assumptions

- Annotation coordinates are treated as LiDAR-like for the prototype because
  BEV support diagnostics were much stronger under the LiDAR assumption.
- The coordinate frame is still not explicitly declared by the dataset.
- The current GroundingDINO 2D prompts are vehicle-focused; pedestrian 2D
  evidence may be absent in the current demo.
- Real detector providers should be implemented as adapters that emit the
  canonical `detector3d` schema.
