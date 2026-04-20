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
