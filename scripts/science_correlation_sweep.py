"""Science experiment: does target correlation dimension correlate with
test accuracy on MNIST?

Runs a grid of hparams. For each combination, trains the MLP on MNIST-5000,
measures the correlation dimension of its projected weight trajectory, and
separately evaluates test accuracy on a held-out 1000-sample subset. Writes
one record per run to a JSON.

This answers the single most important open question in the FractalTrainer
v1 design: the target dim of 1.5 was chosen heuristically — is there any
real relationship between a trajectory's fractal dim and the model's
generalization?

Grid:
    learning_rate: [0.005, 0.01, 0.03, 0.1]
    weight_decay:  [0.0, 0.01]
    optimizer:     ["sgd", "adam"]
    seeds:         [42, 101, 2024]
    → 4 × 2 × 2 × 3 = 48 runs, ~5-6 min compute.

Usage:
    python scripts/science_correlation_sweep.py \\
        [--out results/science_correlation_raw.json]
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


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(64, 32), output_dim=10,
                 dropout=0.0):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def _mnist_loaders(train_size: int, test_size: int, batch_size: int,
                   data_dir: str, seed: int):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    train_full = datasets.MNIST(data_dir, train=True, download=True,
                                transform=transform)
    test_full = datasets.MNIST(data_dir, train=False, download=True,
                               transform=transform)

    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(train_full), size=train_size, replace=False)
    test_idx = rng.choice(len(test_full), size=test_size, replace=False)

    train_loader = DataLoader(Subset(train_full, train_idx.tolist()),
                              batch_size=batch_size, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(Subset(test_full, test_idx.tolist()),
                             batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _evaluate(model: nn.Module, loader) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += float(F.cross_entropy(logits, y).item())
            n_batches += 1
    return correct / max(total, 1), loss_sum / max(n_batches, 1)


def _run_one(hparams: dict, n_steps: int, snapshot_every: int,
             train_size: int, test_size: int, data_dir: str, out_dir: str,
             run_id: str) -> dict:
    model = MLP(input_dim=784, hidden_dims=(64, 32), output_dim=10,
                dropout=float(hparams.get("dropout", 0.0)))

    # Use the hparam seed to both split MNIST and seed the trainer so the
    # same (hparams) pair reproduces.
    train_loader, test_loader = _mnist_loaders(
        train_size=train_size, test_size=test_size,
        batch_size=int(hparams["batch_size"]),
        data_dir=data_dir, seed=int(hparams["init_seed"]),
    )

    trainer = InstrumentedTrainer(
        model=model, dataloader=train_loader, loss_fn=F.cross_entropy,
        hparams=hparams, snapshot_every=snapshot_every,
        out_dir=out_dir, run_id=run_id,
    )
    t0 = time.time()
    run = trainer.train(n_steps)
    train_wall_s = time.time() - t0

    trajectory = np.load(run.snapshot_path)
    projected = project_random(trajectory, n_components=16, seed=0)
    dim_res = correlation_dim(projected, seed=0)
    traj_m = trajectory_metrics(projected)

    # Evaluate on held-out test
    test_acc, test_loss = _evaluate(model, test_loader)

    return {
        "run_id": run_id,
        "hparams": hparams,
        "n_steps": n_steps,
        "n_snapshots": run.n_snapshots,
        "final_loss": run.final_loss,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "train_wall_s": train_wall_s,
        "correlation_dim": dim_res.dim if np.isfinite(dim_res.dim) else None,
        "correlation_dim_r_squared": dim_res.r_squared if np.isfinite(
            dim_res.r_squared) else None,
        "correlation_dim_error": dim_res.error,
        "trajectory_metrics": traj_m,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Science: dim vs test_acc sweep")
    parser.add_argument("--out", type=str,
                        default="results/science_correlation_raw.json")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--trajectory-dir", type=str,
                        default="results/science_trajectories")
    parser.add_argument("--lrs", type=float, nargs="+",
                        default=[0.005, 0.01, 0.03, 0.1])
    parser.add_argument("--weight-decays", type=float, nargs="+",
                        default=[0.0, 0.01])
    parser.add_argument("--optimizers", type=str, nargs="+",
                        default=["sgd", "adam"])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 101, 2024])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args(argv)

    grid = list(itertools.product(
        args.lrs, args.weight_decays, args.optimizers, args.seeds))
    n_runs = len(grid)
    print(f"[science] grid size: {n_runs} runs "
          f"(lr={len(args.lrs)} × wd={len(args.weight_decays)} × "
          f"opt={len(args.optimizers)} × seeds={len(args.seeds)})")

    records: list[dict] = []
    t_start = time.time()
    for i, (lr, wd, opt, seed) in enumerate(grid, start=1):
        hp = {
            "learning_rate": float(lr),
            "batch_size": int(args.batch_size),
            "weight_decay": float(wd),
            "dropout": float(args.dropout),
            "init_seed": int(seed),
            "optimizer": opt,
        }
        run_id = f"science_lr{lr}_wd{wd}_{opt}_seed{seed}".replace(".", "p")
        print(f"[science] {i}/{n_runs}  lr={lr}  wd={wd}  opt={opt}  "
              f"seed={seed}")
        try:
            rec = _run_one(
                hp, n_steps=args.n_steps,
                snapshot_every=args.snapshot_every,
                train_size=args.train_size, test_size=args.test_size,
                data_dir=args.data_dir, out_dir=args.trajectory_dir,
                run_id=run_id,
            )
        except Exception as e:
            rec = {"run_id": run_id, "hparams": hp, "error": f"{type(e).__name__}: {e}"}
        records.append(rec)
        if "error" not in rec:
            print(f"           dim={rec.get('correlation_dim')}  "
                  f"test_acc={rec.get('test_accuracy'):.4f}  "
                  f"wall={rec.get('train_wall_s'):.1f}s")
        else:
            print(f"           ERROR: {rec['error']}")

    elapsed = time.time() - t_start
    print(f"[science] total wall clock: {elapsed:.1f}s "
          f"({elapsed / n_runs:.1f}s/run average)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "grid": {
                "learning_rates": args.lrs,
                "weight_decays": args.weight_decays,
                "optimizers": args.optimizers,
                "seeds": args.seeds,
                "batch_size": args.batch_size,
                "dropout": args.dropout,
            },
            "n_runs": n_runs,
            "n_steps": args.n_steps,
            "snapshot_every": args.snapshot_every,
            "total_wall_s": elapsed,
            "records": records,
        }, f, indent=2, default=str)

    print(f"[science] raw results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
