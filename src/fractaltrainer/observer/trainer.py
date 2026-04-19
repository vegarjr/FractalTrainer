"""InstrumentedTrainer — runs a PyTorch training loop and captures weight snapshots."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fractaltrainer.observer.snapshot import StreamWriter, WeightSnapshot


@dataclass
class TrajectoryRun:
    run_id: str
    hparams: dict
    n_steps: int
    loss_history: list[float]
    snapshot_path: str
    meta_path: str
    final_loss: float
    wall_clock_s: float
    snapshot_overhead_s: float
    n_snapshots: int


def _flatten_weights(model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.data.flatten().cpu() for p in model.parameters()]).numpy()


def _build_optimizer(model: nn.Module, hparams: dict) -> torch.optim.Optimizer:
    name = hparams.get("optimizer", "adam").lower()
    lr = float(hparams["learning_rate"])
    wd = float(hparams.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer: {name!r}")


class InstrumentedTrainer:
    """Runs a PyTorch training loop and emits weight snapshots.

    The model, dataloader, and loss function are injected by the caller so
    this class stays agnostic to model architecture and task.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: Iterable,
        loss_fn: Any,
        hparams: dict,
        snapshot_every: int = 20,
        out_dir: str = "results/trajectories",
        run_id: str | None = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.hparams = dict(hparams)
        self.snapshot_every = int(snapshot_every)
        self.out_dir = Path(out_dir)
        self.run_id = run_id or f"run_{int(time.time())}"

        seed = int(hparams.get("init_seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, n_steps: int) -> TrajectoryRun:
        writer = StreamWriter(str(self.out_dir), self.run_id)
        optimizer = _build_optimizer(self.model, self.hparams)

        # Capture step-0 snapshot before any training
        loss_history: list[float] = []
        snapshot_overhead_s = 0.0

        t0 = time.time()
        t_snap = time.time()
        writer.append(
            WeightSnapshot(
                step=0, flat_weights=_flatten_weights(self.model), loss=float("nan")
            )
        )
        snapshot_overhead_s += time.time() - t_snap

        self.model.train()
        iterator = iter(self.dataloader)
        step = 0
        while step < n_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.dataloader)
                batch = next(iterator)

            x, y = batch
            optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            step += 1
            loss_history.append(float(loss.item()))

            if step % self.snapshot_every == 0 or step == n_steps:
                t_snap = time.time()
                writer.append(
                    WeightSnapshot(
                        step=step,
                        flat_weights=_flatten_weights(self.model),
                        loss=float(loss.item()),
                    )
                )
                snapshot_overhead_s += time.time() - t_snap

        wall_clock_s = time.time() - t0

        extra_meta = {
            "hparams": self.hparams,
            "n_steps": n_steps,
            "snapshot_every": self.snapshot_every,
            "wall_clock_s": wall_clock_s,
            "snapshot_overhead_s": snapshot_overhead_s,
            "snapshot_overhead_frac": (
                snapshot_overhead_s / wall_clock_s if wall_clock_s > 0 else 0.0
            ),
            "final_loss": float(loss_history[-1]) if loss_history else float("nan"),
            "loss_history": loss_history,
        }
        meta = writer.finalize(extra_meta=extra_meta)

        return TrajectoryRun(
            run_id=self.run_id,
            hparams=self.hparams,
            n_steps=n_steps,
            loss_history=loss_history,
            snapshot_path=str(writer.traj_path),
            meta_path=str(writer.meta_path),
            final_loss=extra_meta["final_loss"],
            wall_clock_s=wall_clock_s,
            snapshot_overhead_s=snapshot_overhead_s,
            n_snapshots=meta["n_snapshots"],
        )
