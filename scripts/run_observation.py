"""Sprint 1 demo: instrument training, emit weight-snapshot trajectory + metrics.

Usage:
    python scripts/run_observation.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

# Make `src/` importable when running this as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.observer.trainer import InstrumentedTrainer  # noqa: E402
from fractaltrainer.geometry.trajectory import trajectory_metrics  # noqa: E402


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
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


def _build_mnist_loaders(
    train_subset_size: int,
    test_subset_size: int,
    batch_size: int,
    data_dir: str,
    seed: int,
):
    from torchvision import datasets, transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    train_ds_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds_full = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(train_ds_full), size=train_subset_size, replace=False)
    test_idx = rng.choice(len(test_ds_full), size=test_subset_size, replace=False)

    train_ds = Subset(train_ds_full, train_idx.tolist())
    test_ds = Subset(test_ds_full, test_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Observe a training run")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--hparams", type=str, default=None,
                        help="override path to hparams.yaml")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    hparams_path = Path(args.hparams or cfg["hparams_path"])
    with open(hparams_path) as f:
        hparams = yaml.safe_load(f)

    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    out_dir = cfg["output"]["out_dir"]

    model = MLP(
        input_dim=model_cfg["input_dim"],
        hidden_dims=model_cfg["hidden_dims"],
        output_dim=model_cfg["output_dim"],
        dropout=float(hparams.get("dropout", 0.0)),
    )

    train_loader, _test_loader = _build_mnist_loaders(
        train_subset_size=dataset_cfg["train_subset_size"],
        test_subset_size=dataset_cfg["test_subset_size"],
        batch_size=int(hparams["batch_size"]),
        data_dir=dataset_cfg["data_dir"],
        seed=int(hparams["init_seed"]),
    )

    run_id = args.run_id or f"obs_{int(time.time())}"

    trainer = InstrumentedTrainer(
        model=model,
        dataloader=train_loader,
        loss_fn=F.cross_entropy,
        hparams=hparams,
        snapshot_every=int(training_cfg["snapshot_every"]),
        out_dir=out_dir,
        run_id=run_id,
    )

    print(f"[observer] run_id={run_id}")
    print(f"[observer] hparams={hparams}")
    print(f"[observer] training {training_cfg['n_steps']} steps "
          f"(snapshots every {training_cfg['snapshot_every']})")

    run = trainer.train(int(training_cfg["n_steps"]))

    trajectory = np.load(run.snapshot_path)
    metrics = trajectory_metrics(trajectory)

    metrics_path = Path(run.meta_path).with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "run_id": run.run_id,
                "hparams": run.hparams,
                "n_snapshots": run.n_snapshots,
                "trajectory_shape": list(trajectory.shape),
                "final_loss": run.final_loss,
                "wall_clock_s": run.wall_clock_s,
                "snapshot_overhead_frac": (
                    run.snapshot_overhead_s / run.wall_clock_s
                    if run.wall_clock_s > 0 else 0.0
                ),
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    print(f"[observer] trajectory saved: {run.snapshot_path}  shape={trajectory.shape}")
    print(f"[observer] metrics saved:    {metrics_path}")
    print(f"[observer] final_loss={run.final_loss:.4f}  wall_clock={run.wall_clock_s:.1f}s "
          f"snapshot_overhead={run.snapshot_overhead_s / run.wall_clock_s * 100:.1f}%")
    print(f"[observer] trajectory_metrics:")
    for k, v in metrics.items():
        print(f"    {k:>24s} = {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
