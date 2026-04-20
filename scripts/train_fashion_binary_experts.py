"""v3 Sprint 8 — Train 10 Fashion-MNIST binary experts (× 3 seeds).

Fashion-MNIST classes:
  0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat,
  5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle-boot

10 binary tasks defined over these labels (class-1 is the named set):

  fashion_upperbody   = {0, 2, 3, 4, 6}  # clothing for the torso
  fashion_footwear    = {5, 7, 9}        # sandal/sneaker/boot
  fashion_even_idx    = {0, 2, 4, 6, 8}  # even-indexed classes
  fashion_first_half  = {0, 1, 2, 3, 4}  # first five classes
  fashion_corners     = {0, 1, 8, 9}     # first-2 + last-2
  fashion_middle      = {3, 4, 5, 6}     # middle four
  fashion_no_bag      = {0, 1, 2, 3, 4, 6, 7, 9}  # everything except bag+sandal
  fashion_warm        = {2, 3, 4, 6}     # covering garments
  fashion_athletic    = {1, 5, 7}        # trouser+sandal+sneaker
  fashion_casual      = {0, 5, 7}        # tshirt+sandal+sneaker

Saves trajectories to results/sprint8_v3_fashion_trajectories/.

Architecture: same MLP (784→64→32→10), same hparams as Sprint 4/7
(lr=0.01, adam, bs=64). 500 steps, 25 snapshots.
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
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.observer.trainer import InstrumentedTrainer  # noqa: E402


FASHION_BINARY_TASKS: dict[str, tuple[int, ...]] = {
    "fashion_upperbody":  (0, 2, 3, 4, 6),
    "fashion_footwear":   (5, 7, 9),
    "fashion_even_idx":   (0, 2, 4, 6, 8),
    "fashion_first_half": (0, 1, 2, 3, 4),
    "fashion_corners":    (0, 1, 8, 9),
    "fashion_middle":     (3, 4, 5, 6),
    "fashion_no_bag":     (0, 1, 2, 3, 4, 6, 7, 9),
    "fashion_warm":       (2, 3, 4, 6),
    "fashion_athletic":   (1, 5, 7),
    "fashion_casual":     (0, 5, 7),
}
SEEDS = [42, 101, 2024]


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


class FashionBinaryDataset(Dataset):
    """y → 1 if y ∈ target_subset else 0."""
    def __init__(self, base, target_subset: set[int]):
        self.base, self.target = base, set(target_subset)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(int(y) in self.target)


def _train_expert(
    name: str, subset: tuple[int, ...], seed: int,
    data_dir: str, out_dir: str,
    n_steps: int = 500, train_size: int = 5000,
    batch_size: int = 64, snapshot_every: int = 10,
) -> tuple[Path, float]:
    from torchvision import datasets, transforms
    # FashionMNIST shares MNIST's normalization empirically; use its
    # own stats is cleaner but the MLP learns to adapt either way.
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST stats
    ])
    base = datasets.FashionMNIST(
        str(Path(data_dir) / "FashionMNIST"),
        train=True, download=True, transform=t,
    )
    ds = FashionBinaryDataset(base, set(subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=train_size, replace=False)
    loader = DataLoader(Subset(ds, idx.tolist()),
                         batch_size=batch_size, shuffle=True, drop_last=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    hparams = {
        "learning_rate": 0.01, "batch_size": batch_size,
        "weight_decay": 0.0, "dropout": 0.0,
        "init_seed": seed, "optimizer": "adam",
    }
    run_id = f"ext_{name}_seed{seed}"
    trainer = InstrumentedTrainer(
        model=model, dataloader=loader, loss_fn=F.cross_entropy,
        hparams=hparams, snapshot_every=snapshot_every,
        out_dir=out_dir, run_id=run_id,
    )
    t0 = time.time()
    run = trainer.train(n_steps)
    elapsed = time.time() - t0
    return Path(run.snapshot_path), elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--out-dir", type=str,
                        default="results/sprint8_v3_fashion_trajectories")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--log-out", type=str,
                        default="results/sprint8_v3_fashion_training_log.json")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fashion binary tasks: {len(FASHION_BINARY_TASKS)}")
    for name, subset in FASHION_BINARY_TASKS.items():
        print(f"  {name:<22s}  y ∈ {set(subset)}")

    log: list[dict] = []
    t_start = time.time()
    total = len(FASHION_BINARY_TASKS) * len(SEEDS)
    i = 0
    for name, subset in FASHION_BINARY_TASKS.items():
        for seed in SEEDS:
            i += 1
            traj_path = out_dir / f"ext_{name}_seed{seed}_trajectory.npy"
            if traj_path.is_file():
                print(f"  [{i:>2}/{total}] {name} seed={seed}  (cached)")
                log.append({"task": name, "seed": seed,
                             "traj_path": str(traj_path),
                             "elapsed_s": 0.0, "skipped": True})
                continue
            path, elapsed = _train_expert(
                name, subset, seed,
                args.data_dir, str(out_dir),
                n_steps=args.n_steps,
            )
            log.append({"task": name, "seed": seed,
                         "traj_path": str(path),
                         "elapsed_s": elapsed, "skipped": False})
            print(f"  [{i:>2}/{total}] {name} seed={seed}  "
                  f"trained in {elapsed:.1f}s")
    total_elapsed = time.time() - t_start
    print(f"Total training wall clock: {total_elapsed:.1f}s "
          f"({total_elapsed / 60:.1f} min)")

    Path(args.log_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_out, "w") as f:
        json.dump({
            "tasks": {k: list(v) for k, v in FASHION_BINARY_TASKS.items()},
            "seeds": SEEDS,
            "total_elapsed_s": total_elapsed,
            "log": log,
        }, f, indent=2, default=str)
    print(f"training log: {args.log_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
