"""Sprint 2 demo: load a trajectory, compare its geometric shape to a target.

Steps:
    1. Load a (n_snapshots, n_params) trajectory .npy.
    2. Optionally project to lower-dim via random projection (default) or PCA.
    3. Compute correlation dim (primary) + box-counting dim (sanity).
    4. Also compute the vendored baseline fractal_summary + trajectory metrics.
    5. Compute divergence score against the target.
    6. Emit a comparison_report.json next to the trajectory.

Usage:
    python scripts/run_comparison.py --trajectory <path> [--target <path>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.geometry.box_counting import box_counting_dim  # noqa: E402
from fractaltrainer.geometry.correlation_dim import correlation_dim  # noqa: E402
from fractaltrainer.geometry.fractal_summary import fractal_metrics  # noqa: E402
from fractaltrainer.geometry.trajectory import trajectory_metrics  # noqa: E402
from fractaltrainer.observer.projector import project_pca, project_random  # noqa: E402
from fractaltrainer.target.divergence import (  # noqa: E402
    divergence_score,
    should_intervene,
    within_band,
)
from fractaltrainer.target.target_shape import TargetShape, load_target  # noqa: E402


def _project(trajectory: np.ndarray, target: TargetShape) -> np.ndarray:
    method = target.projection.method
    if method == "none":
        return trajectory
    n_components = target.projection.n_components
    if trajectory.shape[1] <= n_components:
        return trajectory
    if method == "random_proj":
        return project_random(trajectory, n_components=n_components,
                              seed=target.projection.seed)
    if method == "pca":
        return project_pca(trajectory, n_components=n_components)
    raise ValueError(f"unknown projection method: {method!r}")


def _as_native(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _as_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_native(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare trajectory to target")
    parser.add_argument("--trajectory", type=str, required=True,
                        help="path to a (n_snapshots, n_params) .npy")
    parser.add_argument("--target", type=str, default="configs/target_shape.yaml")
    parser.add_argument("--report", type=str, default=None,
                        help="output report path; default beside trajectory")
    args = parser.parse_args(argv)

    traj_path = Path(args.trajectory)
    trajectory = np.load(traj_path)
    target = load_target(args.target)

    projected = _project(trajectory, target)

    primary_method = target.method
    if primary_method == "correlation_dim":
        dim_res = correlation_dim(projected, seed=target.projection.seed)
    elif primary_method == "box_counting":
        dim_res = box_counting_dim(projected)
    else:
        raise ValueError(f"unknown target method: {primary_method!r}")

    current_dim = float(dim_res.dim)
    score = divergence_score(current_dim, target)
    intervene = should_intervene(score, target)
    in_band = within_band(current_dim, target)

    # Sanity: always also run box-counting on a 3-D projection for cross-check
    if trajectory.shape[1] > 3:
        proj3 = project_random(trajectory, n_components=3,
                               seed=target.projection.seed)
    else:
        proj3 = trajectory
    bc_sanity = box_counting_dim(proj3)

    baseline = {
        "trajectory_metrics": trajectory_metrics(projected),
        "fractal_summary": fractal_metrics(projected),
    }

    report = {
        "trajectory_path": str(traj_path),
        "trajectory_shape": list(trajectory.shape),
        "projection": {
            "method": target.projection.method,
            "n_components": target.projection.n_components,
            "seed": target.projection.seed,
            "projected_shape": list(projected.shape),
        },
        "target": target.to_dict(),
        "primary_result": {
            "method": primary_method,
            "dim": current_dim,
            **{k: v for k, v in dim_res.to_dict().items() if k != "dim"},
        },
        "sanity_box_counting_3d": bc_sanity.to_dict(),
        "divergence": {
            "score": score,
            "within_band": in_band,
            "should_intervene": intervene,
            "band": [target.band_low, target.band_high],
        },
        "baseline_metrics": baseline,
    }
    report = _as_native(report)

    out_path = Path(args.report) if args.report else traj_path.with_suffix(
        ".comparison_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[comparator] trajectory: {traj_path}  shape={trajectory.shape}")
    print(f"[comparator] projected shape: {projected.shape}")
    print(f"[comparator] primary ({primary_method}): dim={current_dim:.3f} "
          f"r_squared={dim_res.r_squared:.3f}")
    print(f"[comparator] sanity (box-counting 3d): dim={bc_sanity.dim:.3f}")
    print(f"[comparator] target: {target.dim_target} ± {target.tolerance} "
          f"=> band [{target.band_low:.2f}, {target.band_high:.2f}]")
    print(f"[comparator] divergence score: {score:.3f}  within_band={in_band}  "
          f"should_intervene={intervene}")
    print(f"[comparator] report saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
