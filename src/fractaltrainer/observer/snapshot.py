"""Weight snapshot capture + streaming to disk."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WeightSnapshot:
    step: int
    flat_weights: np.ndarray
    loss: float

    def to_meta(self) -> dict:
        return {
            "step": self.step,
            "loss": self.loss,
            "n_params": int(self.flat_weights.shape[0]),
        }


class StreamWriter:
    """Accumulates WeightSnapshots and flushes them to a single .npy file.

    Usage:
        writer = StreamWriter(out_dir, run_id)
        writer.append(snap)
        ...
        writer.finalize()  # returns path to the (N, D) .npy + metadata json
    """

    def __init__(self, out_dir: str, run_id: str, buffer_size: int = 10):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.buffer_size = buffer_size
        self._buffer: list[WeightSnapshot] = []
        self._finalized = False
        self._all_snapshots: list[WeightSnapshot] = []

    @property
    def traj_path(self) -> Path:
        return self.out_dir / f"{self.run_id}_trajectory.npy"

    @property
    def meta_path(self) -> Path:
        return self.out_dir / f"{self.run_id}_meta.json"

    def append(self, snap: WeightSnapshot) -> None:
        if self._finalized:
            raise RuntimeError("StreamWriter already finalized")
        self._buffer.append(snap)
        self._all_snapshots.append(snap)
        if len(self._buffer) >= self.buffer_size:
            self._buffer = []

    def finalize(self, extra_meta: dict[str, Any] | None = None) -> dict:
        """Write full trajectory (N, D) .npy and metadata JSON. Return paths + summary."""
        if self._finalized:
            raise RuntimeError("StreamWriter already finalized")
        if not self._all_snapshots:
            raise RuntimeError("No snapshots to finalize")

        trajectory = np.stack([s.flat_weights for s in self._all_snapshots], axis=0)
        np.save(self.traj_path, trajectory)

        meta = {
            "run_id": self.run_id,
            "n_snapshots": len(self._all_snapshots),
            "n_params": int(trajectory.shape[1]),
            "trajectory_shape": list(trajectory.shape),
            "trajectory_path": str(self.traj_path),
            "snapshots": [s.to_meta() for s in self._all_snapshots],
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        self._finalized = True
        self._buffer = []
        return meta
