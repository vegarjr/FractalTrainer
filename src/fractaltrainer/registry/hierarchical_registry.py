"""HierarchicalRegistry — dataset-level coarse routing above task-level fine.

Design rationale (v3 Sprint 8):

The flat FractalRegistry does O(N) nearest-neighbor search in signature
space. At small N this is fine (0.36 ms at N=93 per Sprint 7). At
N ≫ 10³ it starts to matter, but more importantly: a flat search can
**semantically cross-contaminate** — a Fashion query might route to
a nearby MNIST task's signature if the dataset-level separation isn't
enforced.

HierarchicalRegistry enforces the separation with a two-level gate:

    Level 1 — dataset coarse routing
        maintain one FractalRegistry per distinct metadata[dataset_key].
        For each dataset, store the mean signature (centroid).
        A query's first hop: pick the dataset whose centroid is
        closest to the query.

    Level 2 — task-level fine routing
        within the chosen dataset's sub-registry, run the same
        flat find_nearest() we already validated in Sprints 3-7.

Payoff: (a) at scale, the O(N) scan shrinks to O(N_dataset) with a
single O(D) centroid-distance computation; (b) cross-dataset
semantic pollution is prevented unless the query's centroid
distance genuinely favors the wrong dataset, in which case the
wrong answer is a failure of the coarse signal itself, not of the
data structure.

The HierarchicalRegistry composes a FractalRegistry rather than
inheriting from one — each dataset gets its own full registry with
unmodified semantics, so the match/compose/spawn decision and the
calibrate_thresholds method still work per-dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from fractaltrainer.registry.fractal_registry import (
    FractalEntry,
    FractalRegistry,
    RetrievalResult,
)


@dataclass
class HierarchicalDecision:
    """Output of HierarchicalRegistry.find_nearest_hierarchical.

    Attributes:
        query_name: the label recorded for this query.
        chosen_dataset: the dataset selected by coarse centroid
            routing — the sub-registry that was searched.
        centroid_distance: L2 distance from the query signature to
            the chosen dataset's centroid.
        all_centroid_distances: {dataset_name: centroid_distance} for
            every registered dataset (for diagnosis).
        retrieval: the within-dataset find_nearest result.
    """

    query_name: str
    chosen_dataset: str
    centroid_distance: float
    all_centroid_distances: dict[str, float]
    retrieval: RetrievalResult

    def to_dict(self) -> dict:
        return {
            "query_name": self.query_name,
            "chosen_dataset": self.chosen_dataset,
            "centroid_distance": float(self.centroid_distance),
            "all_centroid_distances": {
                k: float(v)
                for k, v in self.all_centroid_distances.items()
            },
            "retrieval": self.retrieval.to_dict(),
        }


class HierarchicalRegistry:
    """Two-level registry: dataset-coarse above task-fine.

    Maintains one FractalRegistry per distinct value of
    `metadata[dataset_key]` plus a per-dataset centroid signature for
    coarse routing.

    Usage:
        h = HierarchicalRegistry(dataset_key="dataset")
        h.add(entry_with_dataset_metadata)
        decision = h.find_nearest_hierarchical(query_signature)
        print(decision.chosen_dataset, decision.retrieval.nearest.name)
    """

    def __init__(self, dataset_key: str = "dataset"):
        self._sub_registries: dict[str, FractalRegistry] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self.dataset_key = dataset_key

    # ── Mutation ──────────────────────────────────────────────────

    def add(self, entry: FractalEntry) -> None:
        dataset = entry.metadata.get(self.dataset_key)
        if dataset is None:
            raise ValueError(
                f"entry {entry.name!r} missing "
                f"metadata[{self.dataset_key!r}] — "
                "HierarchicalRegistry requires a dataset label on every entry")
        if dataset not in self._sub_registries:
            self._sub_registries[dataset] = FractalRegistry()
        self._sub_registries[dataset].add(entry)
        self._recompute_centroid(dataset)

    def remove(self, name: str) -> FractalEntry | None:
        """Remove by entry name. Scans all sub-registries."""
        for dataset, sub in self._sub_registries.items():
            if name in sub:
                removed = sub.remove(name)
                if len(sub) > 0:
                    self._recompute_centroid(dataset)
                else:
                    del self._centroids[dataset]
                    del self._sub_registries[dataset]
                return removed
        return None

    def _recompute_centroid(self, dataset: str) -> None:
        entries = self._sub_registries[dataset].entries()
        stacked = np.stack([e.signature for e in entries])
        self._centroids[dataset] = stacked.mean(axis=0)

    def __len__(self) -> int:
        return sum(len(r) for r in self._sub_registries.values())

    def __contains__(self, name: str) -> bool:
        return any(name in r for r in self._sub_registries.values())

    def datasets(self) -> list[str]:
        return list(self._sub_registries.keys())

    def sub_registry(self, dataset: str) -> FractalRegistry | None:
        return self._sub_registries.get(dataset)

    def centroid(self, dataset: str) -> np.ndarray | None:
        return self._centroids.get(dataset)

    def all_entries(self) -> list[FractalEntry]:
        """Flat view across all datasets."""
        out: list[FractalEntry] = []
        for sub in self._sub_registries.values():
            out.extend(sub.entries())
        return out

    # ── Query ─────────────────────────────────────────────────────

    def _coarse_route(
        self, query_signature: np.ndarray,
    ) -> tuple[str, dict[str, float]]:
        if not self._centroids:
            raise ValueError(
                "HierarchicalRegistry is empty — cannot route")
        if not isinstance(query_signature, np.ndarray):
            raise TypeError("query_signature must be np.ndarray")
        centroid_distances = {
            ds: float(np.linalg.norm(c - query_signature))
            for ds, c in self._centroids.items()
        }
        # dict ordering is insertion-order since 3.7, so ties resolve
        # deterministically to the first-inserted dataset.
        best_dataset = min(centroid_distances, key=centroid_distances.get)
        return best_dataset, centroid_distances

    def find_nearest_hierarchical(
        self,
        query_signature: np.ndarray,
        k: int = 1,
        query_name: str = "query",
        exclude: Iterable[str] = (),
    ) -> HierarchicalDecision:
        """Two-phase nearest neighbor: dataset then task.

        Args:
            query_signature: (D,) float array.
            k: number of task-level nearest to return within the
                chosen dataset.
            query_name: label recorded on the retrieval result.
            exclude: entry names to skip in the task-level search.

        Returns:
            HierarchicalDecision recording the chosen dataset, all
            centroid distances (for diagnosis), and the within-dataset
            retrieval result.

        Raises:
            ValueError: if the registry is empty.
        """
        if len(self) == 0:
            raise ValueError(
                "HierarchicalRegistry is empty — cannot find_nearest")
        chosen_dataset, centroid_distances = self._coarse_route(query_signature)
        sub = self._sub_registries[chosen_dataset]
        retrieval = sub.find_nearest(
            query_signature, k=k, query_name=query_name, exclude=exclude,
        )
        return HierarchicalDecision(
            query_name=query_name,
            chosen_dataset=chosen_dataset,
            centroid_distance=centroid_distances[chosen_dataset],
            all_centroid_distances=centroid_distances,
            retrieval=retrieval,
        )
