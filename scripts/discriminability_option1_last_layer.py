"""v3 exploration — Option 1: signature on the last-layer-only trajectory.

The original discriminability experiment (on all 52,650 parameters of the
MLP) gave a ratio of 0.86× — the 9-d signature was dominated by task-
invariant feature-learning noise and could not discriminate tasks.

Option 1 hypothesis: task-specific signal lives in the classification head
(the last Linear layer's 330 parameters, 0.6% of the model). Slicing the
trajectory to just those 330 dims should amplify task signal and shrink
init-noise signal.

We re-use the trajectories saved by the first experiment — no retraining.

Parameter layout (verified from nn.Sequential weight ordering):
    [0     : 50176] Linear(784,64).weight
    [50176 : 50240] Linear(784,64).bias
    [50240 : 52288] Linear(64,32).weight
    [52288 : 52320] Linear(64,32).bias
    [52320 : 52640] Linear(32,10).weight    ← last layer starts here
    [52640 : 52650] Linear(32,10).bias
Last layer total: indices [52320:52650] = 330 params.

Two subexperiments:
    (a) last-layer trajectory projected to 16-d via random projection
        (same pipeline as the all-params version, just on fewer dims)
    (b) last-layer trajectory raw (no projection) — 330-d is small enough
        that GP correlation dim should work directly
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
from fractaltrainer.target.golden_run import (  # noqa: E402
    SIGNATURE_FEATURES,
    _build_signature,
    _signature_to_vector,
)


# Parameter layout for 784→64→32→10 MLP with two ReLUs between Linears.
LAST_LAYER_START = 50240 + 2048 + 32  # = 52320
LAST_LAYER_END = LAST_LAYER_START + 32 * 10 + 10  # = 52650


def _z_distance(sig_a: dict, sig_b: dict, epsilon: float = 1e-6) -> float:
    va = _signature_to_vector(sig_a)
    vb = _signature_to_vector(sig_b)
    scale = (np.abs(va) + np.abs(vb)) / 2.0 + epsilon
    z = (va - vb) / scale
    mask = np.isfinite(z)
    if not mask.any():
        return float("inf")
    return float(np.linalg.norm(z[mask]))


def _signature_from_full_trajectory(traj: np.ndarray, mode: str,
                                     projection_seed: int = 0) -> dict:
    if mode == "full_projected":
        # Original pipeline: project all 52,650 to 16-d
        projected = project_random(traj, n_components=16, seed=projection_seed)
    elif mode == "last_layer_projected":
        # Slice to last-layer params, then project to 16-d
        last = traj[:, LAST_LAYER_START:LAST_LAYER_END]
        projected = project_random(last, n_components=16,
                                    seed=projection_seed)
    elif mode == "last_layer_raw":
        # Slice to last-layer params, no projection (330-d raw)
        projected = traj[:, LAST_LAYER_START:LAST_LAYER_END]
    else:
        raise ValueError(f"unknown mode: {mode}")
    dim_res = correlation_dim(projected, seed=projection_seed)
    traj_m = trajectory_metrics(projected)
    return _build_signature(float(dim_res.dim) if np.isfinite(dim_res.dim)
                             else float("nan"), traj_m)


def _load_trajectories(traj_dir: Path, tasks: list[str], seeds: list[int]
                        ) -> dict[tuple[str, int], np.ndarray]:
    """Load the 6 trajectory files from the first discriminability experiment."""
    out: dict[tuple[str, int], np.ndarray] = {}
    for task in tasks:
        for seed in seeds:
            p = traj_dir / f"discrim_{task}_seed{seed}_trajectory.npy"
            if not p.is_file():
                raise FileNotFoundError(f"trajectory not found: {p}")
            out[(task, seed)] = np.load(p)
    return out


def _run_mode(mode: str, trajectories: dict[tuple[str, int], np.ndarray]
              ) -> dict:
    signatures: dict[tuple[str, int], dict] = {}
    for key, traj in trajectories.items():
        signatures[key] = _signature_from_full_trajectory(traj, mode=mode)

    keys = list(signatures.keys())
    pair_table: list[dict] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            d = _z_distance(signatures[a], signatures[b])
            pair_table.append({
                "a_task": a[0], "a_seed": a[1],
                "b_task": b[0], "b_seed": b[1],
                "same_task": a[0] == b[0],
                "distance": d,
            })

    same = [p["distance"] for p in pair_table if p["same_task"]]
    diff = [p["distance"] for p in pair_table if not p["same_task"]]

    # Filter out inf/nan for means (can happen at very small scaling ranges)
    same_finite = [d for d in same if np.isfinite(d)]
    diff_finite = [d for d in diff if np.isfinite(d)]
    mean_same = float(np.mean(same_finite)) if same_finite else float("nan")
    mean_diff = float(np.mean(diff_finite)) if diff_finite else float("nan")
    ratio = mean_diff / mean_same if mean_same > 0 else float("inf")

    return {
        "mode": mode,
        "signatures": {f"{t}_seed{s}": sig
                        for (t, s), sig in signatures.items()},
        "pairwise": pair_table,
        "n_same": len(same),
        "n_diff": len(diff),
        "mean_same_task_distance": mean_same,
        "mean_diff_task_distance": mean_diff,
        "ratio": ratio,
    }


def _print_results(label: str, result: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"  MODE: {label}")
    print("=" * 72)
    print(f"{'A':<26s}  {'B':<26s}  dist  same?")
    print("-" * 72)
    for p in result["pairwise"]:
        a = f"({p['a_task']}, {p['a_seed']})"
        b = f"({p['b_task']}, {p['b_seed']})"
        flag = "yes" if p["same_task"] else "no"
        d = p["distance"]
        d_str = f"{d:5.2f}" if np.isfinite(d) else "  inf"
        print(f"{a:<26s}  {b:<26s}  {d_str}  {flag}")
    print()
    print(f"  mean same-task (different seeds): {result['mean_same_task_distance']:.3f}  "
          f"(n={result['n_same']})")
    print(f"  mean different-task:              {result['mean_diff_task_distance']:.3f}  "
          f"(n={result['n_diff']})")
    print(f"  ratio (diff / same):              {result['ratio']:.2f}")


def _verdict(ratio: float) -> list[str]:
    lines: list[str] = []
    if ratio > 2.0:
        lines.append(f"DISCRIMINATIVE: ratio {ratio:.2f}× — task identity "
                      "is readable from signature.")
        lines.append("→ Mixture-of-Fractals routing is buildable at this "
                      "level of abstraction.")
    elif ratio > 1.3:
        lines.append(f"WEAKLY DISCRIMINATIVE: ratio {ratio:.2f}× — "
                      "signal exists but is modest; routing will be noisy.")
    else:
        lines.append(f"NOT DISCRIMINATIVE: ratio {ratio:.2f}×")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str,
                        default="results/discriminability_trajectories")
    parser.add_argument("--out", type=str,
                        default="results/discriminability_option1_last_layer.json")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["digit_class", "parity", "high_vs_low"])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 101])
    args = parser.parse_args(argv)

    traj_dir = Path(args.traj_dir)
    trajectories = _load_trajectories(traj_dir, args.tasks, args.seeds)
    print(f"loaded {len(trajectories)} trajectories from {traj_dir}")
    shape = next(iter(trajectories.values())).shape
    print(f"trajectory shape per run: {shape}  "
          f"(n_snapshots={shape[0]}, n_params={shape[1]})")
    print(f"last-layer slice: indices "
          f"[{LAST_LAYER_START}, {LAST_LAYER_END}) → "
          f"{LAST_LAYER_END - LAST_LAYER_START} params")

    results = {}
    for mode, label in [
        ("full_projected",        "baseline (full weights → 16-d proj)"),
        ("last_layer_projected",  "last layer only → 16-d proj"),
        ("last_layer_raw",        "last layer only (raw 330-d)"),
    ]:
        r = _run_mode(mode, trajectories)
        _print_results(label, r)
        verdict = _verdict(r["ratio"])
        for v in verdict:
            print(f"  {v}")
        results[mode] = r
        results[mode]["verdict"] = verdict

    # Summary table
    print("\n" + "=" * 72)
    print("  SUMMARY — ratio by mode (higher = more task-discriminative)")
    print("=" * 72)
    for mode, r in results.items():
        print(f"  {mode:<28s}  ratio = {r['ratio']:5.2f}  "
              f"(same {r['mean_same_task_distance']:.2f}, "
              f"diff {r['mean_diff_task_distance']:.2f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "last_layer_start": LAST_LAYER_START,
            "last_layer_end": LAST_LAYER_END,
            "tasks": args.tasks,
            "seeds": args.seeds,
            "modes": results,
        }, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
