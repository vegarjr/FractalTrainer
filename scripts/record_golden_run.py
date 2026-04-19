"""Record a golden run: take a known-good training trajectory, measure
its 9-d signature, and save it as a GoldenRun JSON.

The v2 Sprint 2 science experiment (Reviews/05) showed that picking a
scalar correlation-dim target a priori is epistemically broken. This
tool inverts the problem: you pick a *training run that produced good
output*, measure what shape it has, and use that as the target.

Inputs:
    --trajectory <path>    a .npy of shape (n_snapshots, n_params)
    --name <label>         free-form name for the golden run
    --test-accuracy <f>    optional; helps document why this run is golden
    --hparams <path>       optional path to the hparams.yaml used
    --output <path>        where to write the JSON (default: golden_runs/<name>.json)

Usage (recording one of the high-test-acc runs from the science sweep):
    python scripts/record_golden_run.py \\
        --trajectory results/science_trajectories/science_lr0p1_wd0p0_sgd_seed101_trajectory.npy \\
        --name sgd_lr0p1_seed101 \\
        --test-accuracy 0.944 \\
        --notes "top run in 48-run science sweep"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.geometry.correlation_dim import correlation_dim  # noqa: E402
from fractaltrainer.geometry.trajectory import trajectory_metrics  # noqa: E402
from fractaltrainer.observer.projector import project_random  # noqa: E402
from fractaltrainer.target.golden_run import GoldenRun  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a golden run")
    parser.add_argument("--trajectory", type=str, required=True,
                        help="(n_snapshots, n_params) .npy trajectory file")
    parser.add_argument("--name", type=str, required=True,
                        help="label for this golden run")
    parser.add_argument("--test-accuracy", type=float, default=None,
                        help="measured test accuracy for this run (provenance)")
    parser.add_argument("--hparams", type=str, default=None,
                        help="optional path to the hparams.yaml used")
    parser.add_argument("--output", type=str, default=None,
                        help="output .json (default golden_runs/<name>.json)")
    parser.add_argument("--n-components", type=int, default=16,
                        help="random-projection dim for correlation dim")
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args(argv)

    traj_path = Path(args.trajectory)
    if not traj_path.is_file():
        print(f"[record] trajectory not found: {traj_path}", file=sys.stderr)
        return 1

    trajectory = np.load(traj_path)
    if trajectory.ndim != 2:
        print(f"[record] expected 2-D trajectory, got {trajectory.shape}",
              file=sys.stderr)
        return 1

    projected = project_random(trajectory,
                                n_components=args.n_components,
                                seed=args.projection_seed)
    dim_res = correlation_dim(projected, seed=args.projection_seed)
    traj_m = trajectory_metrics(projected)

    hparams_dict = None
    if args.hparams:
        try:
            import yaml
            with open(args.hparams) as f:
                hparams_dict = yaml.safe_load(f)
        except Exception as e:
            print(f"[record] could not load hparams: {e}", file=sys.stderr)

    golden = GoldenRun.from_measurements(
        name=args.name,
        correlation_dim_value=float(dim_res.dim) if np.isfinite(dim_res.dim)
                              else float("nan"),
        trajectory_metrics_dict=traj_m,
        test_accuracy=args.test_accuracy,
        hparams=hparams_dict,
        source_trajectory=str(traj_path),
        notes=args.notes,
    )

    output = Path(args.output) if args.output else (
        REPO_ROOT / "golden_runs" / f"{args.name}.json")
    golden.save(output)

    print(f"[record] golden run saved: {output}")
    print(f"[record] trajectory shape: {trajectory.shape}  projected: {projected.shape}")
    print(f"[record] signature:")
    for k, v in golden.signature.items():
        print(f"    {k:>24s} = {v}")
    if args.test_accuracy is not None:
        print(f"[record] provenance: test_accuracy = {args.test_accuracy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
