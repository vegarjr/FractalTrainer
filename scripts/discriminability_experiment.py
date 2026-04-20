"""v3 exploration — Fractal Discriminability Experiment.

Does the 9-d golden-run geometric signature carry task-level information?
Can we tell "model trained on MNIST digit classification" apart from
"model trained on MNIST parity" from the weight trajectory alone?

If yes → the signature is task-discriminative → a registry of fractals
routed by geometric signature is buildable.

If no → the signature is coarse; task routing would need a richer
descriptor (activation statistics, per-layer spectra, etc.).

Three tasks, same data distribution (MNIST images), different labels:
    A. digit_class   — 10-class digit classification (0-9)
    B. parity        — binary (even=0, odd=1), padded to 10-class output
    C. high_vs_low   — binary (0-4 = 0, 5-9 = 1), padded to 10-class

Same architecture (784→64→32→10 MLP), same hparams, same MNIST subset.
Only labels differ. All outputs are 10-class so parameter counts match.

Two seeds per task → 6 trained models → 6 signatures. Compute pairwise
9-d z-normalized distances. Test:

    mean(different-task, same-seed) / mean(same-task, different-seed)

If this ratio >> 1, signatures are task-discriminative.
If ≈ 1, they're not (and the vision needs richer features).
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
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.geometry.correlation_dim import correlation_dim  # noqa: E402
from fractaltrainer.geometry.trajectory import trajectory_metrics  # noqa: E402
from fractaltrainer.observer.projector import project_random  # noqa: E402
from fractaltrainer.observer.trainer import InstrumentedTrainer  # noqa: E402
from fractaltrainer.target.golden_run import (  # noqa: E402
    SIGNATURE_FEATURES,
    _build_signature,
    _signature_to_vector,
)


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(64, 32), output_dim=10):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def _relabel(y: torch.Tensor, task: str) -> torch.Tensor:
    """Map digit labels to task-specific binary labels (padded to 10-dim for
    cross-entropy; CE will only see classes 0 and 1). For 'digit_class',
    labels unchanged.
    """
    if task == "digit_class":
        return y
    elif task == "parity":
        return (y % 2).long()
    elif task == "high_vs_low":
        return (y >= 5).long()
    raise ValueError(f"unknown task: {task}")


class RelabelingDataset(torch.utils.data.Dataset):
    def __init__(self, base, task: str):
        self.base = base
        self.task = task

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, _relabel(torch.tensor(y), self.task).item()


def _mnist_subset(task: str, train_size: int, batch_size: int,
                   data_dir: str, seed: int) -> DataLoader:
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    base = datasets.MNIST(data_dir, train=True, download=True,
                          transform=transform)
    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(base), size=train_size, replace=False)
    relabeled = RelabelingDataset(base, task)
    return DataLoader(Subset(relabeled, train_idx.tolist()),
                      batch_size=batch_size, shuffle=True, drop_last=True)


def _train_and_signature(task: str, seed: int, n_steps: int,
                          train_size: int, batch_size: int,
                          snapshot_every: int, data_dir: str,
                          out_dir: str, projection_seed: int
                          ) -> tuple[dict, float]:
    hparams = {
        "learning_rate": 0.01,
        "batch_size": batch_size,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "init_seed": seed,
        "optimizer": "adam",
    }
    model = MLP()
    loader = _mnist_subset(task, train_size, batch_size,
                           data_dir=data_dir, seed=seed)

    run_id = f"discrim_{task}_seed{seed}"
    trainer = InstrumentedTrainer(
        model=model, dataloader=loader, loss_fn=F.cross_entropy,
        hparams=hparams, snapshot_every=snapshot_every,
        out_dir=out_dir, run_id=run_id,
    )
    t0 = time.time()
    run = trainer.train(n_steps)
    trajectory = np.load(run.snapshot_path)
    projected = project_random(trajectory, n_components=16,
                                seed=projection_seed)
    dim_res = correlation_dim(projected, seed=projection_seed)
    traj_m = trajectory_metrics(projected)
    sig = _build_signature(float(dim_res.dim) if np.isfinite(dim_res.dim)
                           else float("nan"), traj_m)
    elapsed = time.time() - t0
    return sig, elapsed


def _z_distance(sig_a: dict, sig_b: dict, epsilon: float = 1e-6) -> float:
    """Same z-normalized distance we use in golden_run_distance, but
    symmetric: scale = mean(|a|, |b|) + eps.
    """
    va = _signature_to_vector(sig_a)
    vb = _signature_to_vector(sig_b)
    scale = (np.abs(va) + np.abs(vb)) / 2.0 + epsilon
    z = (va - vb) / scale
    mask = np.isfinite(z)
    if not mask.any():
        return float("inf")
    return float(np.linalg.norm(z[mask]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                        default="results/discriminability_experiment.json")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--snapshot-every", type=int, default=10)  # 50 snapshots
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--traj-dir", type=str,
                        default="results/discriminability_trajectories")
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101])
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["digit_class", "parity", "high_vs_low"])
    args = parser.parse_args(argv)

    combos = list(itertools.product(args.tasks, args.seeds))
    print(f"Training {len(combos)} models ({len(args.tasks)} tasks × "
          f"{len(args.seeds)} seeds)")
    print(f"Architecture: 784→64→32→10 MLP, Adam lr=0.01, "
          f"batch={args.batch_size}")
    print(f"{args.n_steps} steps, snapshot every {args.snapshot_every} "
          f"→ {args.n_steps // args.snapshot_every + 1} snapshots per run")

    signatures: dict[tuple[str, int], dict] = {}
    timings: list[dict] = []
    for task, seed in combos:
        print(f"\n[{task:>15s}, seed={seed}] training...")
        sig, elapsed = _train_and_signature(
            task=task, seed=seed, n_steps=args.n_steps,
            train_size=args.train_size, batch_size=args.batch_size,
            snapshot_every=args.snapshot_every,
            data_dir=args.data_dir, out_dir=args.traj_dir,
            projection_seed=args.projection_seed,
        )
        signatures[(task, seed)] = sig
        timings.append({"task": task, "seed": seed, "elapsed_s": elapsed})
        print(f"    elapsed {elapsed:.1f}s  "
              f"corr_dim={sig['correlation_dim']:.3f}  "
              f"path_length={sig['total_path_length']:.3f}  "
              f"tortuosity={sig['tortuosity']:.3f}")

    # Pairwise distances
    keys = list(signatures.keys())
    pair_table: list[dict] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            d = _z_distance(signatures[a], signatures[b])
            same_task = a[0] == b[0]
            pair_table.append({
                "a_task": a[0], "a_seed": a[1],
                "b_task": b[0], "b_seed": b[1],
                "same_task": same_task,
                "distance": d,
            })

    same_task_dists = [p["distance"] for p in pair_table if p["same_task"]]
    diff_task_dists = [p["distance"] for p in pair_table if not p["same_task"]]

    print()
    print("=" * 72)
    print("PAIRWISE DISTANCES (z-normalized L2 in 9-d signature space)")
    print("=" * 72)
    print(f"{'A (task, seed)':<30s}  {'B (task, seed)':<30s}  dist  same-task?")
    print("-" * 72)
    for p in pair_table:
        a = f"({p['a_task']}, {p['a_seed']})"
        b = f"({p['b_task']}, {p['b_seed']})"
        flag = "yes" if p["same_task"] else "no"
        print(f"{a:<30s}  {b:<30s}  {p['distance']:5.2f}  {flag}")

    mean_same = float(np.mean(same_task_dists)) if same_task_dists else float("nan")
    mean_diff = float(np.mean(diff_task_dists)) if diff_task_dists else float("nan")
    ratio = mean_diff / mean_same if mean_same > 0 else float("inf")

    print()
    print(f"mean distance, same task (different seeds):     {mean_same:.3f}  "
          f"(n={len(same_task_dists)})")
    print(f"mean distance, different task (any seed):       {mean_diff:.3f}  "
          f"(n={len(diff_task_dists)})")
    print(f"ratio (different/same):                         {ratio:.2f}")

    print()
    print("=" * 72)
    verdict: list[str] = []
    if ratio > 2.0:
        verdict.append(f"DISCRIMINATIVE: ratio {ratio:.2f}× — "
                       "different-task signatures are meaningfully "
                       "separable from same-task.")
        verdict.append("→ The 9-d signature CAN be used to route between "
                       "fractals based on task. The Mixture-of-Fractals "
                       "vision is implementable on existing machinery.")
    elif ratio > 1.3:
        verdict.append(f"WEAKLY DISCRIMINATIVE: ratio {ratio:.2f}× — signal "
                       "exists but is modest.")
        verdict.append("→ Routing may work but will be noisy. Extended "
                       "signatures (per-layer, activation statistics) would "
                       "likely help.")
    else:
        verdict.append(f"NOT DISCRIMINATIVE: ratio {ratio:.2f}× — "
                       "within-task variance ≈ between-task variance.")
        verdict.append("→ The 9-d signature does NOT carry usable task "
                       "information at this scale. Registry routing would "
                       "need a richer descriptor first.")
    for line in verdict:
        print(line)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n_tasks": len(args.tasks),
            "n_seeds_per_task": len(args.seeds),
            "n_steps": args.n_steps,
            "train_size": args.train_size,
            "signatures": {f"{t}_seed{s}": sig for (t, s), sig in signatures.items()},
            "pairwise": pair_table,
            "mean_same_task_distance": mean_same,
            "mean_diff_task_distance": mean_diff,
            "ratio_diff_over_same": ratio,
            "verdict": verdict,
            "timings": timings,
        }, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
