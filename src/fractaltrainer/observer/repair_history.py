"""RepairHistoryReader — the meta-observer.

In v1 the observer watches the AI's *training* trajectory (weight snapshots
over SGD steps). v2 adds a second observer: the trajectory of the repair
loop itself. Each accepted hparam patch moves the AI's hyperparameter
state from one point in hparam-space to another; the sequence of those
points is a *meta-trajectory*.

The shape of this meta-trajectory answers: "does the AI rewrite itself
fractally?" Same geometric machinery as v1 (trajectory_metrics,
correlation_dim) can be applied once hparam states are embedded in R^d.

Embedding choice (6-d real vector per hparam state):
    dim 0: log10(learning_rate)                in ~[-6, 0]
    dim 1: log2(batch_size)                    in  [0, 10]
    dim 2: weight_decay                        in  [0, 0.1]
    dim 3: dropout                             in  [0, 0.9]
    dim 4: init_seed / (2^31 - 1)              in  [0, 1]
    dim 5: optimizer_index in {0,1,2}          in  {0, 1, 2}   (sgd, adam, adamw)

The optimizer index is discrete — the trajectory can "jump" in that dim
when the LLM switches optimizer. That's fine; trajectory_metrics handles
it. correlation_dim will see a small N and return NaN below min_points,
which is honest.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


OPTIMIZER_INDEX: dict[str, int] = {"sgd": 0, "adam": 1, "adamw": 2}
_INT_31_MAX = float(2 ** 31 - 1)


@dataclass
class MetaTrajectory:
    """A sequence of hparam states extracted from a repair log."""

    states: np.ndarray              # shape (n_points, 6) — embedded hparam vectors
    hparams_sequence: list[dict]    # raw hparam dicts, same ordering as states
    iterations: list[int]           # iteration number per point
    statuses: list[str]             # status string per transition (len = n_points - 1)
    divergence_scores: list[float]  # divergence score AT each state (len = n_points)
    sources: list[Path] = field(default_factory=list)  # which log file(s) contributed

    @property
    def n_points(self) -> int:
        return int(self.states.shape[0])

    @property
    def n_transitions(self) -> int:
        return max(0, self.n_points - 1)


def embed_hparams(hparams: dict) -> np.ndarray:
    """Map a single hparam dict → (6,) float vector."""
    lr = float(hparams.get("learning_rate", 1e-3))
    lr = max(lr, 1e-12)
    bs = int(hparams.get("batch_size", 1))
    bs = max(bs, 1)
    wd = float(hparams.get("weight_decay", 0.0))
    do = float(hparams.get("dropout", 0.0))
    seed = int(hparams.get("init_seed", 0))
    opt = str(hparams.get("optimizer", "adam")).lower()
    opt_idx = OPTIMIZER_INDEX.get(opt, 1)  # default "adam"

    return np.array([
        math.log10(lr),
        math.log2(bs),
        wd,
        do,
        seed / _INT_31_MAX,
        float(opt_idx),
    ], dtype=np.float64)


def _valid_entry(entry: dict) -> bool:
    """Does this log entry have the fields we need to place it on a trajectory?"""
    if not isinstance(entry, dict):
        return False
    if "hparams_before" not in entry:
        return False
    return isinstance(entry["hparams_before"], dict)


class RepairHistoryReader:
    """Parses one or more repair_log.jsonl files into a MetaTrajectory.

    Rules:
      - The first point in the trajectory is the hparams_before of the
        first valid log entry.
      - For each subsequent entry, we extend the trajectory with
        hparams_after if status indicates the patch was kept
        (``accepted``) AND hparams_after differs from hparams_before.
        An ``accepted`` entry where hparams_after == hparams_before
        (e.g. "already within band") does not add a new point.
      - Rollback / error / no-fix entries do not add a new point —
        those patches were reverted (or never applied), so the hparam
        state did not change.
      - Divergence score is recorded per point: divergence_before at
        the start, divergence_after at each point added by an accept.

    If multiple log files are passed, they're concatenated in the given
    order (useful for comparing multiple closed-loop runs).
    """

    def __init__(self):
        pass

    def read(self, paths: str | Path | list[str | Path]) -> MetaTrajectory:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        path_list = [Path(p) for p in paths]

        all_entries: list[dict] = []
        for p in path_list:
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not all_entries:
            return MetaTrajectory(
                states=np.zeros((0, 6)),
                hparams_sequence=[],
                iterations=[],
                statuses=[],
                divergence_scores=[],
                sources=path_list,
            )

        valid = [e for e in all_entries if _valid_entry(e)]
        if not valid:
            return MetaTrajectory(
                states=np.zeros((0, 6)),
                hparams_sequence=[],
                iterations=[],
                statuses=[],
                divergence_scores=[],
                sources=path_list,
            )

        # First point: hparams_before of the first valid entry.
        first = valid[0]
        hparams_sequence: list[dict] = [dict(first["hparams_before"])]
        iterations: list[int] = [int(first.get("iteration", 0))]
        divergences: list[float] = [self._safe_float(first.get("divergence_before"))]
        statuses: list[str] = []

        for entry in valid:
            status = str(entry.get("status", "")).lower()
            if status != "accepted":
                continue
            hp_after = entry.get("hparams_after")
            if not isinstance(hp_after, dict):
                continue
            if hp_after == hparams_sequence[-1]:
                continue
            hparams_sequence.append(dict(hp_after))
            iterations.append(int(entry.get("iteration", 0)))
            divergences.append(self._safe_float(entry.get("divergence_after")))
            statuses.append(status)

        states = np.stack([embed_hparams(h) for h in hparams_sequence], axis=0)

        return MetaTrajectory(
            states=states,
            hparams_sequence=hparams_sequence,
            iterations=iterations,
            statuses=statuses,
            divergence_scores=divergences,
            sources=path_list,
        )

    @staticmethod
    def _safe_float(v) -> float:
        try:
            x = float(v)
            return x if math.isfinite(x) else float("inf")
        except (TypeError, ValueError):
            return float("inf")
