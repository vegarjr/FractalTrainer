"""v3 pre-experiments: scale robustness + cross-dataset.

Pre-experiment A (scale): does the 2.3× ratio from Option 1 hold when
we scale from 3 tasks × 2 seeds → 10 tasks × 3 seeds (from 3 task pairs
to 45)?

Pre-experiment B (cross-dataset): does the signature distinguish
DIFFERENT DATA DISTRIBUTIONS too (MNIST vs Fashion-MNIST), or only
different labels within a distribution?

Design:
    10 MNIST-relabeled tasks + 1 Fashion-MNIST task (10-way)
    3 seeds each → 33 trained models
    Same architecture (784→64→32→10 MLP), same hparams.

Signature mode:
    "last_layer_projected" — the mode that won in Option 1 (ratio 2.27).
    Random-project the 330 last-layer params of each trajectory to 16-d,
    compute 9-d signature.

Reports three ratios:
    - within-task (same labels, different seeds)
    - cross-task, same-dataset (MNIST relabelings)
    - cross-dataset (MNIST vs Fashion-MNIST)

Cached trajectories from the first 6-model run are reused.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.geometry.correlation_dim import correlation_dim  # noqa: E402
from fractaltrainer.geometry.trajectory import trajectory_metrics  # noqa: E402
from fractaltrainer.observer.projector import project_random  # noqa: E402
from fractaltrainer.observer.trainer import InstrumentedTrainer  # noqa: E402
from fractaltrainer.target.golden_run import (  # noqa: E402
    _build_signature,
    _signature_to_vector,
)


LAST_LAYER_START = 50240 + 2048 + 32  # = 52320
LAST_LAYER_END = LAST_LAYER_START + 32 * 10 + 10  # = 52650


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


def _relabel_mnist(y: int, task: str) -> int:
    if task == "digit_class":
        return int(y)
    if task == "parity":
        return int(y % 2)
    if task == "high_vs_low":
        return int(y >= 5)
    if task == "mod3":
        return int(y % 3)
    if task == "mod5":
        return int(y % 5)
    if task == "primes_vs_rest":
        return int(y in (2, 3, 5, 7))
    if task == "ones_vs_teens":
        return int(y <= 4)
    if task == "triangular":
        return int(y in (1, 3, 6))
    if task == "fibonacci":
        return int(y in (1, 2, 3, 5, 8))
    if task == "middle_456":
        return int(y in (4, 5, 6))
    raise ValueError(f"unknown mnist task: {task}")


class RelabeledMNIST(Dataset):
    def __init__(self, base, task: str):
        self.base = base
        self.task = task

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, _relabel_mnist(int(y), self.task)


def _make_loader(task: str, train_size: int, batch_size: int,
                  data_dir: str, seed: int) -> DataLoader:
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    if task == "fashion_class":
        # Fashion-MNIST standard 10-way classification
        base = datasets.FashionMNIST(
            str(data_dir_path / "FashionMNIST"),
            train=True, download=True, transform=transform,
        )
        ds = base
    else:
        base = datasets.MNIST(data_dir, train=True, download=True,
                              transform=transform)
        ds = RelabeledMNIST(base, task)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=train_size, replace=False)
    return DataLoader(Subset(ds, idx.tolist()),
                      batch_size=batch_size, shuffle=True, drop_last=True)


def _train_one(task: str, seed: int, n_steps: int, train_size: int,
                batch_size: int, snapshot_every: int, data_dir: str,
                out_dir: str) -> tuple[np.ndarray, float]:
    """Train one model and return its trajectory + wall clock."""
    run_id = f"ext_{task}_seed{seed}"
    traj_path = Path(out_dir) / f"{run_id}_trajectory.npy"
    if traj_path.is_file():
        # Reuse cached
        return np.load(traj_path), 0.0

    hparams = {
        "learning_rate": 0.01, "batch_size": batch_size,
        "weight_decay": 0.0, "dropout": 0.0,
        "init_seed": seed, "optimizer": "adam",
    }
    model = MLP()
    loader = _make_loader(task, train_size, batch_size, data_dir, seed)
    trainer = InstrumentedTrainer(
        model=model, dataloader=loader, loss_fn=F.cross_entropy,
        hparams=hparams, snapshot_every=snapshot_every,
        out_dir=out_dir, run_id=run_id,
    )
    t0 = time.time()
    run = trainer.train(n_steps)
    elapsed = time.time() - t0
    return np.load(run.snapshot_path), elapsed


def _last_layer_signature(trajectory: np.ndarray, projection_seed: int = 0
                           ) -> dict:
    last = trajectory[:, LAST_LAYER_START:LAST_LAYER_END]
    projected = project_random(last, n_components=16, seed=projection_seed)
    dim_res = correlation_dim(projected, seed=projection_seed)
    traj_m = trajectory_metrics(projected)
    return _build_signature(
        float(dim_res.dim) if np.isfinite(dim_res.dim) else float("nan"),
        traj_m,
    )


def _z_distance(sig_a: dict, sig_b: dict, eps: float = 1e-6) -> float:
    va = _signature_to_vector(sig_a)
    vb = _signature_to_vector(sig_b)
    scale = (np.abs(va) + np.abs(vb)) / 2.0 + eps
    z = (va - vb) / scale
    mask = np.isfinite(z)
    if not mask.any():
        return float("inf")
    return float(np.linalg.norm(z[mask]))


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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024])
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--out-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--results-out", type=str,
                        default="results/discriminability_extended.json")
    parser.add_argument("--projection-seed", type=int, default=0)
    args = parser.parse_args(argv)

    combos = list(itertools.product(ALL_TASKS, args.seeds))
    print(f"Total models to train: {len(combos)} "
          f"({len(ALL_TASKS)} tasks × {len(args.seeds)} seeds)")
    print(f"  MNIST relabeled tasks: {len(MNIST_TASKS)}")
    print(f"  Fashion-MNIST tasks:   {len(FASHION_TASKS)}")

    signatures: dict[tuple[str, int], dict] = {}
    timings: list[dict] = []
    for i, (task, seed) in enumerate(combos, start=1):
        print(f"\n[{i:2d}/{len(combos)}] {task:<20s} seed={seed}")
        traj, elapsed = _train_one(
            task=task, seed=seed, n_steps=args.n_steps,
            train_size=args.train_size, batch_size=args.batch_size,
            snapshot_every=args.snapshot_every,
            data_dir=args.data_dir, out_dir=args.out_dir,
        )
        sig = _last_layer_signature(traj, args.projection_seed)
        signatures[(task, seed)] = sig
        timings.append({"task": task, "seed": seed, "elapsed_s": elapsed})
        cached = elapsed == 0.0
        print(f"    {'CACHED' if cached else f'trained in {elapsed:.1f}s'}  "
              f"corr_dim={sig['correlation_dim']:.3f}  "
              f"path_len={sig['total_path_length']:.3f}  "
              f"tort={sig['tortuosity']:.3f}")

    # Pairwise distances
    keys = list(signatures.keys())
    pair_table: list[dict] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            d = _z_distance(signatures[a], signatures[b])
            pair_table.append({
                "a_task": a[0], "a_seed": a[1],
                "b_task": b[0], "b_seed": b[1],
                "a_dataset": _task_dataset(a[0]),
                "b_dataset": _task_dataset(b[0]),
                "same_task": a[0] == b[0],
                "same_dataset": _task_dataset(a[0]) == _task_dataset(b[0]),
                "distance": d,
            })

    # Partition the pairs
    within_task = [p for p in pair_table if p["same_task"]]
    cross_task_within_dataset = [
        p for p in pair_table
        if not p["same_task"] and p["same_dataset"]
    ]
    cross_dataset = [p for p in pair_table if not p["same_dataset"]]

    def _mean_finite(pairs):
        xs = [p["distance"] for p in pairs if np.isfinite(p["distance"])]
        return float(np.mean(xs)) if xs else float("nan")

    def _std_finite(pairs):
        xs = [p["distance"] for p in pairs if np.isfinite(p["distance"])]
        return float(np.std(xs)) if xs else float("nan")

    mean_within = _mean_finite(within_task)
    mean_cross_within_ds = _mean_finite(cross_task_within_dataset)
    mean_cross_ds = _mean_finite(cross_dataset)
    std_within = _std_finite(within_task)
    std_cross_within_ds = _std_finite(cross_task_within_dataset)
    std_cross_ds = _std_finite(cross_dataset)

    ratio_task = (mean_cross_within_ds / mean_within
                   if mean_within > 0 else float("inf"))
    ratio_ds = (mean_cross_ds / mean_within
                 if mean_within > 0 else float("inf"))
    ratio_ds_vs_task = (mean_cross_ds / mean_cross_within_ds
                         if mean_cross_within_ds > 0 else float("inf"))

    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"within-task (same labels, diff seeds):   "
          f"{mean_within:.3f} ± {std_within:.3f}  (n={len(within_task)})")
    print(f"cross-task, same-dataset (MNIST):         "
          f"{mean_cross_within_ds:.3f} ± {std_cross_within_ds:.3f}  "
          f"(n={len(cross_task_within_dataset)})")
    print(f"cross-dataset (MNIST vs Fashion):         "
          f"{mean_cross_ds:.3f} ± {std_cross_ds:.3f}  (n={len(cross_dataset)})")
    print()
    print(f"  ratio cross-task / within-task:         {ratio_task:.2f}×")
    print(f"  ratio cross-dataset / within-task:      {ratio_ds:.2f}×")
    print(f"  ratio cross-dataset / cross-task:       {ratio_ds_vs_task:.2f}×")

    # Per-task-pair breakdown for MNIST tasks (mean distance between each pair of tasks)
    print()
    print("  Per-task-pair means (MNIST relabelings, averaged over seeds):")
    task_pair_means: dict[tuple[str, str], list[float]] = {}
    for p in pair_table:
        a, b = p["a_task"], p["b_task"]
        if p["a_dataset"] == "mnist" and p["b_dataset"] == "mnist":
            key = tuple(sorted((a, b)))
            task_pair_means.setdefault(key, []).append(p["distance"])
    rows = sorted(
        ((k, float(np.mean([v for v in vs if np.isfinite(v)])))
         for k, vs in task_pair_means.items()),
        key=lambda x: x[1],
    )
    for (a, b), mean_d in rows[:20]:
        flag = "(same)" if a == b else ""
        print(f"    {a:<18s} ↔ {b:<18s}  {mean_d:.3f}  {flag}")

    verdict = _verdict(ratio_task, ratio_ds, ratio_ds_vs_task,
                        len(within_task))

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
            "tasks": ALL_TASKS,
            "seeds": args.seeds,
            "n_models": len(combos),
            "signatures": {f"{t}_seed{s}": sig
                            for (t, s), sig in signatures.items()},
            "pairwise": pair_table,
            "stats": {
                "within_task": {"mean": mean_within, "std": std_within,
                                 "n": len(within_task)},
                "cross_task_within_dataset":
                    {"mean": mean_cross_within_ds,
                     "std": std_cross_within_ds,
                     "n": len(cross_task_within_dataset)},
                "cross_dataset":
                    {"mean": mean_cross_ds,
                     "std": std_cross_ds,
                     "n": len(cross_dataset)},
                "ratio_cross_task_over_within": ratio_task,
                "ratio_cross_dataset_over_within": ratio_ds,
                "ratio_cross_dataset_over_cross_task": ratio_ds_vs_task,
            },
            "verdict": verdict,
            "timings": timings,
        }, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


def _verdict(ratio_task: float, ratio_ds: float, ratio_ds_vs_task: float,
              n_within: int) -> list[str]:
    out: list[str] = []
    if ratio_task > 2.0:
        out.append(f"SCALE-ROBUST: ratio cross-task/within-task = "
                    f"{ratio_task:.2f}× (Option 1's 2.27× holds at scale).")
        out.append("→ Mixture-of-Fractals routing within a dataset is "
                    "empirically viable.")
    elif ratio_task > 1.3:
        out.append(f"Task signal weakens at scale: "
                    f"{ratio_task:.2f}× (down from Option 1's 2.27×).")
        out.append("→ Routing will be noisier with more tasks; "
                    "consider richer signatures.")
    else:
        out.append(f"Task signal breaks at scale: "
                    f"{ratio_task:.2f}× (Option 1 finding did not "
                    "generalize).")
        out.append("→ Need a richer descriptor (activation-based) before "
                    "building the registry.")

    if ratio_ds > ratio_task * 1.5:
        out.append(f"CROSS-DATASET is much stronger: {ratio_ds:.2f}× vs "
                    f"{ratio_task:.2f}× for cross-task. Dataset identity "
                    "is even more readable from signature than task "
                    "identity is.")
        out.append("→ The registry naturally organizes by dataset first, "
                    "task second. Hierarchical routing makes sense.")
    elif ratio_ds > 1.3:
        out.append(f"Cross-dataset distinguishable but not dominant: "
                    f"{ratio_ds:.2f}×. Task vs dataset signal are "
                    "comparable.")
    else:
        out.append(f"Cross-dataset NOT distinguishable: "
                    f"{ratio_ds:.2f}×. The last-layer signature captures "
                    "label-structure, not input-distribution.")

    return out


if __name__ == "__main__":
    raise SystemExit(main())
