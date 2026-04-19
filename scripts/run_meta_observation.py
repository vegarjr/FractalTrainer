"""v2 demo: observe the geometric shape of the repair loop itself.

The v1 observer watches a training run and emits a weight trajectory.
The v2 meta-observer watches the *repair loop* (across one or more
closed-loop runs) and emits a meta-trajectory: the sequence of hparam
states the AI has mutated itself through.

If the meta-trajectory across many repair runs has a fractal shape, that
is literal recursive self-similarity: the rewriter's rewrites form the
same kind of geometric object the training trajectory does.

Usage:
    python scripts/run_meta_observation.py \\
        --logs results/repair_log.jsonl \\
        [--output results/meta_trajectory.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.geometry.meta_trajectory import summarize_meta_trajectory  # noqa: E402
from fractaltrainer.observer.repair_history import RepairHistoryReader  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the geometric shape of the repair loop itself")
    parser.add_argument("--logs", type=str, nargs="+",
                        default=["results/repair_log.jsonl"],
                        help="one or more repair_log.jsonl paths")
    parser.add_argument("--output", type=str, default=None,
                        help="optional path to write a meta_trajectory.json")
    parser.add_argument("--min-points-for-dim", type=int, default=20,
                        help=("minimum n_points for correlation dim to be "
                              "computed; below this dim is reported as None "
                              "(N is typically small for repair loops — "
                              "concatenate multiple runs to get meaningful dim)"))
    args = parser.parse_args(argv)

    reader = RepairHistoryReader()
    mt = reader.read(args.logs)

    print(f"[meta] logs: {[str(p) for p in mt.sources]}")
    print(f"[meta] meta-trajectory points: {mt.n_points}")
    print(f"[meta] transitions (accepted patches with state change): "
          f"{mt.n_transitions}")

    if mt.n_points == 0:
        print("[meta] no valid log entries found")
        return 1

    summary = summarize_meta_trajectory(
        mt, min_points_for_dim=args.min_points_for_dim)

    print("\n[meta] hparam sequence:")
    for i, (h, it) in enumerate(zip(mt.hparams_sequence, mt.iterations)):
        div = mt.divergence_scores[i] if i < len(mt.divergence_scores) else None
        div_str = f"{div:.3f}" if div is not None and div != float("inf") else "?"
        print(f"  [{i}] iter={it}  div={div_str}  "
              f"lr={h.get('learning_rate')}  bs={h.get('batch_size')}  "
              f"wd={h.get('weight_decay')}  do={h.get('dropout')}  "
              f"opt={h.get('optimizer')}")

    if summary.geometry:
        print("\n[meta] geometry (trajectory_metrics on 6-d hparam embedding):")
        for k, v in summary.geometry.items():
            print(f"  {k:>24s} = {v}")

    if summary.correlation_dim:
        print("\n[meta] correlation dimension:")
        print(f"  dim       = {summary.correlation_dim.dim:.4f}")
        print(f"  r_squared = {summary.correlation_dim.r_squared:.4f}")
        if summary.correlation_dim.error:
            print(f"  error     = {summary.correlation_dim.error}")
    else:
        print(f"\n[meta] correlation dim skipped "
              f"(n_points={mt.n_points} < min_points_for_dim="
              f"{args.min_points_for_dim})")
        print("      to get a meaningful dim, concatenate multiple logs:")
        print("        --logs run1.jsonl run2.jsonl run3.jsonl ...")

    print("\n[meta] convergence signature:")
    for k, v in summary.convergence.items():
        print(f"  {k:>24s} = {v}")

    if args.output:
        out_path = Path(args.output)
        payload = {
            "sources": [str(p) for p in mt.sources],
            "n_points": mt.n_points,
            "n_transitions": mt.n_transitions,
            "iterations": mt.iterations,
            "hparams_sequence": mt.hparams_sequence,
            "divergence_scores": [
                d if d != float("inf") else None for d in mt.divergence_scores
            ],
            "states_shape": list(mt.states.shape),
            "summary": summary.to_dict(),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\n[meta] meta-trajectory report saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
