"""v3 pre-experiment Option 2 — activation signature on a fixed probe batch.

Previous results:
    full_projected weight traj (N=3 tasks): ratio 0.86  (not discriminative)
    last_layer_projected (N=3 tasks):       ratio 2.27  (discriminative)
    last_layer_projected (N=10 tasks):      ratio 1.28  (signal degraded at scale)

The weight-trajectory signature hits a scaling ceiling. Option 2 tries a
fundamentally different descriptor: the model's OUTPUT BEHAVIOR on a
fixed probe batch. If two models solve the same task, their class-
probability distributions on the same inputs should be similar. If
different tasks, they should differ — directly, without needing to
extract signal from a noisy trajectory.

Method:
    1. Load a fixed 100-image MNIST probe batch (seed=12345).
    2. For each of the 33 cached trained models (from the extended
       experiment), reconstruct the MLP from the final snapshot weights,
       run the probe batch through, record softmax outputs.
    3. Activation signature = flatten (100, 10) softmax matrix → 1000-d.
    4. Pairwise distances = L2 on signatures.
    5. Report within-task / cross-task / cross-dataset ratios.

The probe is drawn from MNIST for all models including Fashion-MNIST-
trained ones. That means Fashion models receive OOD input — the weirdness
of their outputs on MNIST is itself a dataset-identity signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset


REPO_ROOT = Path(__file__).resolve().parent.parent


# Same layout as discriminability_extended.py
LAYER_SHAPES = [
    ((64, 784), "net.0.weight"),
    ((64,),     "net.0.bias"),
    ((32, 64),  "net.2.weight"),
    ((32,),     "net.2.bias"),
    ((10, 32),  "net.4.weight"),
    ((10,),     "net.4.bias"),
]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def _load_model_from_final_snapshot(traj_path: Path) -> nn.Module:
    trajectory = np.load(traj_path)
    final_flat = trajectory[-1]  # final training step's weights

    model = MLP()
    offset = 0
    state_dict = {}
    for shape, name in LAYER_SHAPES:
        size = int(np.prod(shape))
        chunk = final_flat[offset:offset + size]
        state_dict[name] = torch.tensor(chunk.reshape(shape),
                                          dtype=torch.float32)
        offset += size
    if offset != final_flat.shape[0]:
        raise RuntimeError(
            f"parameter count mismatch: offset={offset} "
            f"but final_flat has {final_flat.shape[0]} elements")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_probe_batch(data_dir: str, n_probe: int,
                        probe_seed: int) -> torch.Tensor:
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=False, download=True,
                          transform=transform)
    rng = np.random.RandomState(probe_seed)
    idx = rng.choice(len(base), size=n_probe, replace=False)
    batch = torch.stack([base[i][0] for i in idx.tolist()], dim=0)
    return batch


def _activation_signature(model: nn.Module, probe: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(probe)
        probs = F.softmax(logits, dim=1)
    return probs.flatten().cpu().numpy()


def _distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.linalg.norm(sig_a - sig_b))


MNIST_TASKS = [
    "digit_class", "parity", "high_vs_low",
    "mod3", "mod5",
    "primes_vs_rest", "ones_vs_teens", "triangular",
    "fibonacci", "middle_456",
]
FASHION_TASKS = ["fashion_class"]
ALL_TASKS = MNIST_TASKS + FASHION_TASKS


def _task_dataset(task: str) -> str:
    return "fashion_mnist" if task in FASHION_TASKS else "mnist"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024])
    parser.add_argument("--results-out", type=str,
                        default="results/discriminability_option2_activations.json")
    args = parser.parse_args(argv)

    traj_dir = Path(args.traj_dir)
    probe = _build_probe_batch(args.data_dir, args.n_probe, args.probe_seed)
    print(f"Probe batch: {args.n_probe} MNIST test images (probe_seed={args.probe_seed})")
    print(f"Trajectory source: {traj_dir}")

    signatures: dict[tuple[str, int], np.ndarray] = {}
    print(f"\nComputing activation signatures for {len(ALL_TASKS)} tasks × "
          f"{len(args.seeds)} seeds = {len(ALL_TASKS) * len(args.seeds)} models")
    for task in ALL_TASKS:
        for seed in args.seeds:
            traj_path = traj_dir / f"ext_{task}_seed{seed}_trajectory.npy"
            if not traj_path.is_file():
                print(f"  SKIP {task:<20s} seed={seed}  (no trajectory)")
                continue
            model = _load_model_from_final_snapshot(traj_path)
            sig = _activation_signature(model, probe)
            signatures[(task, seed)] = sig
    print(f"Loaded {len(signatures)} signatures")

    # Pairwise distances
    keys = list(signatures.keys())
    pair_table: list[dict] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            d = _distance(signatures[a], signatures[b])
            pair_table.append({
                "a_task": a[0], "a_seed": a[1],
                "b_task": b[0], "b_seed": b[1],
                "a_dataset": _task_dataset(a[0]),
                "b_dataset": _task_dataset(b[0]),
                "same_task": a[0] == b[0],
                "same_dataset": _task_dataset(a[0]) == _task_dataset(b[0]),
                "distance": d,
            })

    within_task = [p for p in pair_table if p["same_task"]]
    cross_task_within_ds = [p for p in pair_table
                             if not p["same_task"] and p["same_dataset"]]
    cross_ds = [p for p in pair_table if not p["same_dataset"]]

    def _stats(pairs):
        xs = [p["distance"] for p in pairs if np.isfinite(p["distance"])]
        if not xs:
            return float("nan"), float("nan"), 0
        return float(np.mean(xs)), float(np.std(xs)), len(xs)

    mean_within, std_within, n_within = _stats(within_task)
    mean_cross_ds_same, std_cross_ds_same, n_cross_ds_same = _stats(cross_task_within_ds)
    mean_cross_ds, std_cross_ds, n_cross_ds = _stats(cross_ds)

    ratio_task = mean_cross_ds_same / mean_within if mean_within > 0 else float("inf")
    ratio_ds = mean_cross_ds / mean_within if mean_within > 0 else float("inf")
    ratio_ds_vs_task = (mean_cross_ds / mean_cross_ds_same
                         if mean_cross_ds_same > 0 else float("inf"))

    print()
    print("=" * 72)
    print("  SUMMARY (activation-based, L2 on 1000-d softmax outputs)")
    print("=" * 72)
    print(f"  within-task (same labels, diff seeds):   "
          f"{mean_within:.4f} ± {std_within:.4f}  (n={n_within})")
    print(f"  cross-task same dataset (MNIST):         "
          f"{mean_cross_ds_same:.4f} ± {std_cross_ds_same:.4f}  "
          f"(n={n_cross_ds_same})")
    print(f"  cross-dataset (MNIST vs Fashion):        "
          f"{mean_cross_ds:.4f} ± {std_cross_ds:.4f}  (n={n_cross_ds})")
    print()
    print(f"  ratio cross-task / within-task:          {ratio_task:.2f}×")
    print(f"  ratio cross-dataset / within-task:       {ratio_ds:.2f}×")
    print(f"  ratio cross-dataset / cross-task:        {ratio_ds_vs_task:.2f}×")

    # Per-task-pair mean distance (within-dataset = MNIST)
    print()
    print("  Per-task-pair mean distances (MNIST relabelings, avg over seeds):")
    task_pairs: dict[tuple[str, str], list[float]] = {}
    for p in pair_table:
        if p["a_dataset"] == "mnist" and p["b_dataset"] == "mnist":
            key = tuple(sorted((p["a_task"], p["b_task"])))
            task_pairs.setdefault(key, []).append(p["distance"])
    rows = sorted(
        ((k, float(np.mean(vs))) for k, vs in task_pairs.items()),
        key=lambda x: x[1],
    )
    for (a, b), mean_d in rows[:24]:
        flag = "(same task)" if a == b else ""
        print(f"    {a:<18s} ↔ {b:<18s}  {mean_d:.4f}  {flag}")

    verdict: list[str] = []
    if ratio_task > 2.0:
        verdict.append(f"DISCRIMINATIVE: cross-task/within-task = "
                        f"{ratio_task:.2f}×  → MoF routing viable.")
    elif ratio_task > 1.3:
        verdict.append(f"WEAKLY DISCRIMINATIVE: "
                        f"{ratio_task:.2f}×  → routing possible but "
                        "noisy at scale.")
    else:
        verdict.append(f"NOT DISCRIMINATIVE: {ratio_task:.2f}×")

    if ratio_ds > ratio_task * 1.5:
        verdict.append(f"Cross-dataset dominates: {ratio_ds:.2f}× vs "
                        f"{ratio_task:.2f}×. Hierarchical dataset → "
                        "task routing recommended.")

    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    for line in verdict:
        print(f"  {line}")

    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n_probe": args.n_probe,
            "probe_seed": args.probe_seed,
            "signature_dim": args.n_probe * 10,
            "n_models": len(signatures),
            "pairwise": pair_table,
            "stats": {
                "within_task":
                    {"mean": mean_within, "std": std_within, "n": n_within},
                "cross_task_within_dataset":
                    {"mean": mean_cross_ds_same, "std": std_cross_ds_same,
                     "n": n_cross_ds_same},
                "cross_dataset":
                    {"mean": mean_cross_ds, "std": std_cross_ds, "n": n_cross_ds},
                "ratio_cross_task_over_within": ratio_task,
                "ratio_cross_dataset_over_within": ratio_ds,
                "ratio_cross_dataset_over_cross_task": ratio_ds_vs_task,
            },
            "verdict": verdict,
        }, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
