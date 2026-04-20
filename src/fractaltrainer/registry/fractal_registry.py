"""FractalRegistry — routed mixture of experts, keyed by activation signature.

Design rationale (from v3 pre-experiments in scripts/discriminability_*.py):

  Attempt 1 (full weight trajectory signature, 9-d):   ratio 0.86 — NO
  Attempt 2 (last-layer weight trajectory, 9-d, N=3):  ratio 2.27 — yes
  Attempt 3 (last-layer weight trajectory, 9-d, N=10): ratio 1.28 — degraded
  Attempt 4 (activation on probe batch, 1000-d):       ratio 3.49 — strong

The registry uses activation signatures. A fixed probe batch (100 MNIST
images) is passed through a trained expert; the flattened 100×10 softmax
output becomes that expert's signature. Signatures cluster tightly by
task (within-task mean distance 3.05) and separate cleanly from other
tasks (cross-task mean 10.66) — a clean gap in [3.61, 7.45].

At query time: run the same probe batch through a new candidate model
(or through a task's data distribution in some other way), compute its
signature, find the k nearest entries in the registry by L2 distance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class FractalEntry:
    """A single registered expert.

    Attributes:
        name: free-form identifier (e.g. "digit_class_seed42").
        signature: (D,) float array — the activation signature on the
            canonical probe batch.
        metadata: free-form dict (task, seed, test_accuracy, model_path, etc.).
    """

    name: str
    signature: np.ndarray
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "signature": self.signature.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FractalEntry":
        return cls(
            name=d["name"],
            signature=np.asarray(d["signature"], dtype=np.float64),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class RetrievalResult:
    """Result of a find_nearest query."""

    query_name: str
    entries: list[FractalEntry]    # sorted near → far
    distances: list[float]         # L2 distances, same order as entries

    @property
    def nearest(self) -> FractalEntry | None:
        return self.entries[0] if self.entries else None

    def to_dict(self) -> dict:
        return {
            "query_name": self.query_name,
            "ranked": [
                {"name": e.name, "distance": float(d),
                 "metadata": e.metadata}
                for e, d in zip(self.entries, self.distances)
            ],
        }


@dataclass
class CalibrationResult:
    """Output of FractalRegistry.calibrate_thresholds.

    Attributes:
        match_threshold: upper-percentile of within-task distances.
        spawn_threshold: lower-percentile of cross-task distances
            (clamped to ≥ match_threshold when the raw percentiles
            cross — see `overlap`).
        within_task_distances: flat array of all within-task pairwise
            L2 distances.
        cross_task_distances: flat array of all cross-task pairwise
            L2 distances.
        within_percentile: upper percentile used on within-task.
        cross_percentile: lower percentile used on cross-task.
        overlap: True when the within and cross distributions overlap
            at the chosen percentiles (raw within_p > raw cross_p);
            the returned thresholds are still valid but the registry
            doesn't have a clean gap between same-task and cross-task.
        n_tasks: number of distinct tasks grouped.
    """

    match_threshold: float
    spawn_threshold: float
    within_task_distances: np.ndarray
    cross_task_distances: np.ndarray
    within_percentile: float
    cross_percentile: float
    overlap: bool
    n_tasks: int

    def to_dict(self) -> dict:
        return {
            "match_threshold": float(self.match_threshold),
            "spawn_threshold": float(self.spawn_threshold),
            "within_percentile": float(self.within_percentile),
            "cross_percentile": float(self.cross_percentile),
            "overlap": bool(self.overlap),
            "n_tasks": int(self.n_tasks),
            "within_distance_stats": {
                "n": int(self.within_task_distances.size),
                "min": float(self.within_task_distances.min()),
                "mean": float(self.within_task_distances.mean()),
                "max": float(self.within_task_distances.max()),
            },
            "cross_distance_stats": {
                "n": int(self.cross_task_distances.size),
                "min": float(self.cross_task_distances.min()),
                "mean": float(self.cross_task_distances.mean()),
                "max": float(self.cross_task_distances.max()),
            },
        }


@dataclass
class GrowthDecision:
    """Routing + growth decision produced by FractalRegistry.decide.

    verdict: one of
        "match"   — min_distance ≤ match_threshold, route to nearest
        "compose" — match_threshold < min_distance ≤ spawn_threshold,
                    blend top-K nearest
        "spawn"   — min_distance > spawn_threshold, no nearby expert,
                    train+register a new fractal
        "empty"   — registry is empty; must spawn the first fractal
    """

    verdict: str
    min_distance: float | None
    retrieval: RetrievalResult | None
    composite_entries: list[FractalEntry] | None = None
    composite_weights: list[float] | None = None

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "min_distance": self.min_distance,
            "retrieval": (self.retrieval.to_dict() if self.retrieval else None),
            "composite_entries": (
                [e.name for e in self.composite_entries]
                if self.composite_entries else None
            ),
            "composite_weights": (
                [float(w) for w in self.composite_weights]
                if self.composite_weights is not None else None
            ),
        }


class FractalRegistry:
    """Routed registry of fractal experts.

    Each entry is a (signature, metadata) pair. Routing = nearest-neighbor
    search in signature space.

    The registry is intentionally simple: no learned gating, no hierarchical
    structure, no ANN index. A naive O(N) linear scan is fine for N ≤ a few
    thousand. If the registry grows past that, swap in a FAISS or ANN index
    behind the same interface.
    """

    def __init__(self, entries: Iterable[FractalEntry] | None = None):
        self._entries: dict[str, FractalEntry] = {}
        if entries:
            for e in entries:
                self.add(e)

    # ── Mutation ──────────────────────────────────────────────────

    def add(self, entry: FractalEntry) -> None:
        """Add an entry. Overwrites silently on duplicate name."""
        if not isinstance(entry.signature, np.ndarray):
            raise TypeError(
                f"signature must be np.ndarray, got {type(entry.signature)}")
        self._entries[entry.name] = entry

    def remove(self, name: str) -> FractalEntry | None:
        return self._entries.pop(name, None)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def entries(self) -> list[FractalEntry]:
        return list(self._entries.values())

    def get(self, name: str) -> FractalEntry | None:
        return self._entries.get(name)

    # ── Query ─────────────────────────────────────────────────────

    def find_nearest(
        self,
        query_signature: np.ndarray,
        k: int = 1,
        query_name: str = "query",
        exclude: Iterable[str] = (),
    ) -> RetrievalResult:
        """Return the k entries with the lowest L2 distance to `query_signature`.

        Args:
            query_signature: (D,) float array.
            k: number of nearest to return (1 ≤ k ≤ len(registry)).
            query_name: label recorded on the RetrievalResult.
            exclude: iterable of entry names to skip (useful for
                leave-one-out evaluation).
        """
        if len(self._entries) == 0:
            return RetrievalResult(query_name=query_name, entries=[], distances=[])
        if not isinstance(query_signature, np.ndarray):
            raise TypeError("query_signature must be np.ndarray")

        exclude_set = set(exclude)
        candidates = [e for e in self._entries.values()
                      if e.name not in exclude_set]
        if not candidates:
            return RetrievalResult(query_name=query_name, entries=[], distances=[])

        dists: list[tuple[float, FractalEntry]] = []
        for e in candidates:
            if e.signature.shape != query_signature.shape:
                continue  # silently skip incompatible-dim entries
            d = float(np.linalg.norm(e.signature - query_signature))
            dists.append((d, e))

        if not dists:
            return RetrievalResult(query_name=query_name, entries=[], distances=[])

        dists.sort(key=lambda t: t[0])
        top = dists[: max(1, min(k, len(dists)))]
        return RetrievalResult(
            query_name=query_name,
            entries=[e for _, e in top],
            distances=[d for d, _ in top],
        )

    def composite_weights(
        self,
        query_signature: np.ndarray,
        k: int = 3,
        temperature: float = 1.0,
        exclude: Iterable[str] = (),
    ) -> tuple[RetrievalResult, np.ndarray]:
        """Find top-k nearest and return inverse-distance weights for composition.

        Returns:
            (RetrievalResult, weights) where weights is a (k,) float array
            summing to 1.0. Softmax over -distance / temperature.

        For use in mixture-of-experts inference: query_signature is the new
        task's probe signature; weights tell you how to combine the k
        nearest experts' predictions.
        """
        res = self.find_nearest(query_signature, k=k, exclude=exclude)
        if not res.distances:
            return res, np.array([])
        # Softmax over negative distances
        d = np.asarray(res.distances, dtype=np.float64)
        # Shift to avoid numeric issues when distances are large
        centered = -(d - d.min()) / max(temperature, 1e-9)
        exp = np.exp(centered)
        weights = exp / exp.sum()
        return res, weights

    # ── Decision ──────────────────────────────────────────────────

    def decide(
        self,
        query_signature: np.ndarray,
        match_threshold: float,
        spawn_threshold: float,
        compose_k: int = 3,
        compose_temperature: float = 1.0,
        exclude: Iterable[str] = (),
    ) -> GrowthDecision:
        """Given a new query signature, return a GrowthDecision.

        match_threshold < spawn_threshold. Typically calibrated from the
        within-task / cross-task distance distribution — see the MVP
        Test A/B/C numbers in Reviews/12.

        Semantics:
            min_distance ≤ match_threshold   →  "match"   (route to nearest)
            match < min ≤ spawn_threshold    →  "compose" (top-K blend)
            min_distance > spawn_threshold   →  "spawn"   (new fractal)
            empty registry                    →  "empty"   (spawn the first)
        """
        if match_threshold > spawn_threshold:
            raise ValueError(
                f"match_threshold ({match_threshold}) must be <= "
                f"spawn_threshold ({spawn_threshold})")
        if len(self._entries) == 0 or len(list(self._entries.keys())) == len(
                list(exclude)) and set(self._entries) == set(exclude):
            return GrowthDecision(verdict="empty", min_distance=None,
                                  retrieval=None)

        res = self.find_nearest(query_signature, k=compose_k, exclude=exclude)
        if not res.distances:
            return GrowthDecision(verdict="empty", min_distance=None,
                                  retrieval=None)

        min_d = res.distances[0]
        if min_d <= match_threshold:
            return GrowthDecision(verdict="match", min_distance=min_d,
                                  retrieval=res)
        if min_d <= spawn_threshold:
            # Compose top-K
            _, weights = self.composite_weights(
                query_signature, k=compose_k,
                temperature=compose_temperature, exclude=exclude,
            )
            return GrowthDecision(
                verdict="compose", min_distance=min_d,
                retrieval=res,
                composite_entries=list(res.entries),
                composite_weights=list(weights),
            )
        return GrowthDecision(verdict="spawn", min_distance=min_d,
                              retrieval=res)

    # ── Calibration ───────────────────────────────────────────────

    def calibrate_thresholds(
        self,
        task_key: str = "task",
        within_percentile: float = 95.0,
        cross_percentile: float = 5.0,
    ) -> CalibrationResult:
        """Compute match/spawn thresholds from the registry's own
        pairwise distance distribution, grouped by metadata[task_key].

        The match_threshold is set at `within_percentile` of the
        within-task distance distribution (so ≥within_percentile of
        same-task queries will land in the match verdict). The
        spawn_threshold is set at `cross_percentile` of the cross-task
        distance distribution (so ≥(100 − cross_percentile)% of
        cross-task queries will land in the spawn verdict).

        Defaults (95, 5) follow the Sprint 3 MVP observation: MNIST
        same-task distances max 3.43, cross-task min 7.32 — 95th of
        within and 5th of cross both land in the (3.43, 7.32) gap,
        reproducing the hand-tuned (5.0, 7.0) without tuning.

        Args:
            task_key: metadata key identifying an entry's task.
            within_percentile: upper percentile of within-task
                distances used for match_threshold. Default 95.
            cross_percentile: lower percentile of cross-task distances
                used for spawn_threshold. Default 5.

        Returns:
            CalibrationResult. When the raw percentiles cross
            (match > spawn), `overlap=True` and spawn_threshold is
            clamped to match_threshold so the caller can still use
            the thresholds (compose band collapses to zero width).

        Raises:
            ValueError: <2 entries, <2 distinct tasks, missing
                task_key on any entry, or no within-task pairs (every
                task has only 1 entry).
        """
        entries = list(self._entries.values())
        if len(entries) < 2:
            raise ValueError(
                "Need at least 2 registered entries to calibrate "
                f"thresholds; have {len(entries)}")

        by_task: dict[str, list[FractalEntry]] = {}
        for e in entries:
            task = e.metadata.get(task_key)
            if task is None:
                raise ValueError(
                    f"Entry {e.name!r} missing metadata[{task_key!r}]; "
                    "cannot separate within/cross-task distances")
            by_task.setdefault(task, []).append(e)

        if len(by_task) < 2:
            raise ValueError(
                f"Need at least 2 distinct tasks; found {len(by_task)} "
                f"under metadata key {task_key!r}")

        within: list[float] = []
        for task_entries in by_task.values():
            for i in range(len(task_entries)):
                for j in range(i + 1, len(task_entries)):
                    d = float(np.linalg.norm(
                        task_entries[i].signature
                        - task_entries[j].signature))
                    within.append(d)
        if not within:
            raise ValueError(
                "No within-task pairs — every task has only 1 entry. "
                "Need ≥2 entries per task for at least one task.")

        cross: list[float] = []
        task_names = list(by_task.keys())
        for i in range(len(task_names)):
            for j in range(i + 1, len(task_names)):
                for a in by_task[task_names[i]]:
                    for b in by_task[task_names[j]]:
                        d = float(np.linalg.norm(a.signature - b.signature))
                        cross.append(d)

        within_arr = np.asarray(within, dtype=np.float64)
        cross_arr = np.asarray(cross, dtype=np.float64)
        match_t = float(np.percentile(within_arr, within_percentile))
        spawn_t = float(np.percentile(cross_arr, cross_percentile))

        overlap = match_t > spawn_t
        if overlap:
            spawn_t = match_t

        return CalibrationResult(
            match_threshold=match_t,
            spawn_threshold=spawn_t,
            within_task_distances=within_arr,
            cross_task_distances=cross_arr,
            within_percentile=within_percentile,
            cross_percentile=cross_percentile,
            overlap=overlap,
            n_tasks=len(by_task),
        )

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "n_entries": len(self._entries),
            "entries": [e.to_dict() for e in self._entries.values()],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "FractalRegistry":
        with open(path) as f:
            payload = json.load(f)
        entries = [FractalEntry.from_dict(e) for e in payload.get("entries", [])]
        return cls(entries=entries)
