"""Reclustering — extract Sprint 12's agglomerative clustering into a reusable function.

Sprint 12 (`scripts/run_learned_anchors_experiment.py`) showed that
agglomerative clustering on Jaccard distances between registered
experts' class-1 label sets recovers the {0-4} / {5-9} anchor split
without being told. This module exposes that logic as a callable the
FractalPipeline can invoke periodically after new spawns.

Two distance metrics are supported:
    - "jaccard" — on metadata["task_labels"] (frozenset of class-1 digits)
    - "signature" — L2 in the registry's signature space (falls back to
                    this when task_labels metadata is missing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from fractaltrainer.registry import FractalEntry, FractalRegistry


@dataclass
class ClusteringResult:
    """Output of `recluster`.

    Attributes:
        clusters: list of clusters; each cluster is a list of entry names.
        anchors: per-cluster union of class-1 label sets (empty frozenset
            when the metric is signature-space).
        metric: which distance metric was used ("jaccard" or "signature").
        k: number of clusters requested.
    """

    clusters: list[list[str]]
    anchors: list[frozenset[int]]
    metric: str
    k: int

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "k": int(self.k),
            "n_clusters": self.n_clusters,
            "clusters": [list(c) for c in self.clusters],
            "anchors": [sorted(a) for a in self.anchors],
        }


def _jaccard(a: frozenset, b: frozenset) -> float:
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def _jaccard_distance_matrix(
    entries: list[FractalEntry], task_label_key: str,
) -> np.ndarray:
    labels: list[frozenset[int]] = []
    for e in entries:
        lbls = e.metadata.get(task_label_key)
        if lbls is None:
            raise ValueError(
                f"entry {e.name!r} missing metadata[{task_label_key!r}]; "
                "use metric='signature' for jaccard-unaware registries"
            )
        labels.append(frozenset(lbls))
    n = len(entries)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = 1.0 - _jaccard(labels[i], labels[j])
    return D


def _signature_distance_matrix(entries: list[FractalEntry]) -> np.ndarray:
    n = len(entries)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(entries[i].signature - entries[j].signature))
            D[i, j] = D[j, i] = d
    return D


def agglomerative_cluster_average_linkage(
    dist_matrix: np.ndarray, k: int,
) -> list[list[int]]:
    """Bottom-up agglomerative clustering with average linkage.

    Lifted from `scripts/run_learned_anchors_experiment.py:147-182`
    verbatim to preserve the Sprint 12 behavior; any fixes here should
    also be applied there.
    """
    n = dist_matrix.shape[0]
    if k >= n:
        return [[i] for i in range(n)]
    clusters: list[list[int]] = [[i] for i in range(n)]
    D = dist_matrix.astype(np.float64).copy()
    while len(clusters) > k:
        m = len(clusters)
        min_d, mi, mj = np.inf, -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                if D[i, j] < min_d:
                    min_d = D[i, j]
                    mi, mj = i, j
        merged = clusters[mi] + clusters[mj]
        del clusters[mj]
        del clusters[mi]
        clusters.append(merged)
        new_m = len(clusters)
        new_D = np.zeros((new_m, new_m))
        for a in range(new_m):
            for b in range(a + 1, new_m):
                ds = [dist_matrix[x, y]
                      for x in clusters[a] for y in clusters[b]]
                new_D[a, b] = new_D[b, a] = float(np.mean(ds))
        D = new_D
    return clusters


def recluster(
    registry: FractalRegistry,
    k: int = 3,
    metric: str = "jaccard",
    task_label_key: str = "task_labels",
) -> ClusteringResult:
    """Partition the registry's entries into `k` clusters.

    Args:
        registry: FractalRegistry with at least k entries.
        k: desired number of clusters.
        metric: "jaccard" (uses metadata[task_label_key] as a
            frozenset) or "signature" (L2 distance on entry.signature).
        task_label_key: metadata key holding the class-1 label set;
            only used when metric="jaccard".

    Returns:
        ClusteringResult.
    """
    if metric not in ("jaccard", "signature"):
        raise ValueError(f"unknown metric: {metric!r}")
    entries = registry.entries()
    if len(entries) < 2:
        return ClusteringResult(
            clusters=[[e.name for e in entries]],
            anchors=[frozenset()], metric=metric, k=k,
        )
    k = max(1, min(k, len(entries)))

    if metric == "jaccard":
        D = _jaccard_distance_matrix(entries, task_label_key)
    else:
        D = _signature_distance_matrix(entries)

    index_clusters = agglomerative_cluster_average_linkage(D, k)

    name_clusters: list[list[str]] = []
    anchor_sets: list[frozenset[int]] = []
    for idx_list in index_clusters:
        cluster_entries = [entries[i] for i in idx_list]
        name_clusters.append([e.name for e in cluster_entries])
        if metric == "jaccard":
            union = set()
            for e in cluster_entries:
                lbls = e.metadata.get(task_label_key)
                if lbls is not None:
                    union |= set(lbls)
            anchor_sets.append(frozenset(union))
        else:
            anchor_sets.append(frozenset())

    return ClusteringResult(
        clusters=name_clusters, anchors=anchor_sets,
        metric=metric, k=k,
    )
