from __future__ import annotations

from pathlib import Path

import numpy as np


def load_lidar_xyz(data_root: Path, lidar_filename: str) -> np.ndarray:
    path = data_root / "middle_lidar" / lidar_filename
    if not path.is_file():
        raise FileNotFoundError(f"点云不存在: {path}")
    z = np.load(path)
    if not all(k in z.files for k in ("x", "y", "z")):
        raise ValueError(f"NPZ 需包含 x,y,z，实际 keys={z.files}")
    return np.stack([z["x"], z["y"], z["z"]], axis=1).astype(np.float32)


def downsample_points(xyz: np.ndarray, max_points: int) -> np.ndarray:
    if xyz.shape[0] <= max_points:
        return xyz
    rng = np.random.default_rng(0)
    idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[idx]
