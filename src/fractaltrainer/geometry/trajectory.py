"""VENDORED from /home/vegar/Documents/Code_Geometry/representations/state_trajectory.py
Source commit: e8a00f05e821f0636d8ac422f49d98c7bb1324ca (Initial commit: Code Geometry Lab MVP)
Original author: Vegar Ratdal
License: MIT (same ownership, vendored for explicit dependency tracking)
Modifications: dropped build_trajectory() (TraceRecord-specific input); stripped
               trace-specific per-feature stats (depth_mean, depth_std,
               function_change_rate) — not meaningful for generic (N, D) arrays.
"""

import numpy as np


def trajectory_metrics(trajectory: np.ndarray) -> dict:
    """Compute geometric metrics on a state trajectory.

    Args:
        trajectory: ndarray of shape (n_points, n_features). Generic over
            feature meaning — works equally on execution-trace features or
            projected neural-network weight snapshots.

    Returns:
        dict with keys: total_path_length, mean_step_norm, step_norm_std,
        dispersion, displacement, tortuosity, mean_curvature, recurrence_rate.
    """
    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory must be 2-D (n_points, n_features), got shape {trajectory.shape}"
        )

    n_points, _ = trajectory.shape
    metrics: dict = {}

    diffs = np.diff(trajectory, axis=0)
    step_norms = np.linalg.norm(diffs, axis=1)

    metrics["total_path_length"] = float(np.sum(step_norms))
    metrics["mean_step_norm"] = (
        float(np.mean(step_norms)) if len(step_norms) > 0 else 0.0
    )
    metrics["step_norm_std"] = (
        float(np.std(step_norms)) if len(step_norms) > 0 else 0.0
    )

    centroid = np.mean(trajectory, axis=0)
    distances_from_centroid = np.linalg.norm(trajectory - centroid, axis=1)
    metrics["dispersion"] = float(np.mean(distances_from_centroid))

    metrics["displacement"] = float(np.linalg.norm(trajectory[-1] - trajectory[0]))

    if metrics["displacement"] > 1e-10:
        metrics["tortuosity"] = metrics["total_path_length"] / metrics["displacement"]
    else:
        metrics["tortuosity"] = float("inf")

    if len(diffs) >= 2:
        angles = []
        for i in range(len(diffs) - 1):
            n1 = np.linalg.norm(diffs[i])
            n2 = np.linalg.norm(diffs[i + 1])
            if n1 > 1e-10 and n2 > 1e-10:
                cos_angle = np.clip(
                    np.dot(diffs[i], diffs[i + 1]) / (n1 * n2), -1.0, 1.0
                )
                angles.append(np.arccos(cos_angle))
        metrics["mean_curvature"] = float(np.mean(angles)) if angles else 0.0
    else:
        metrics["mean_curvature"] = 0.0

    if n_points > 10:
        threshold = np.median(step_norms) * 0.5 if len(step_norms) > 0 else 0.01
        check_indices = list(range(10, n_points, max(1, n_points // 50)))
        recurrence_count = 0
        for i in check_indices:
            dists = np.linalg.norm(trajectory[: max(1, i - 5)] - trajectory[i], axis=1)
            if np.any(dists < threshold):
                recurrence_count += 1
        metrics["recurrence_rate"] = (
            recurrence_count / len(check_indices) if check_indices else 0.0
        )
    else:
        metrics["recurrence_rate"] = 0.0

    return metrics
