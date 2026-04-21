"""Unit tests for integration.recluster."""

from __future__ import annotations

import numpy as np
import pytest

from fractaltrainer.integration.recluster import (
    ClusteringResult,
    agglomerative_cluster_average_linkage,
    recluster,
)
from fractaltrainer.registry import FractalEntry, FractalRegistry


def _entry(name: str, labels: frozenset[int], sig_dim: int = 1000,
           sig_seed: int = 0) -> FractalEntry:
    rng = np.random.RandomState(sig_seed)
    sig = rng.randn(sig_dim).astype(np.float64)
    return FractalEntry(
        name=name, signature=sig,
        metadata={"task": name, "task_labels": labels, "seed": sig_seed},
    )


def test_agglomerative_returns_k_clusters():
    # 6 points; 3 clusters requested
    D = np.array([
        [0, 1, 1, 9, 9, 9],
        [1, 0, 1, 9, 9, 9],
        [1, 1, 0, 9, 9, 9],
        [9, 9, 9, 0, 1, 1],
        [9, 9, 9, 1, 0, 1],
        [9, 9, 9, 1, 1, 0],
    ], dtype=float)
    clusters = agglomerative_cluster_average_linkage(D, k=2)
    assert len(clusters) == 2
    # Two clean clusters
    sets = [set(c) for c in clusters]
    assert {0, 1, 2} in sets
    assert {3, 4, 5} in sets


def test_recluster_jaccard_matches_label_structure():
    reg = FractalRegistry()
    # Low-digit cluster
    reg.add(_entry("low_A", frozenset({0, 1, 2}), sig_seed=1))
    reg.add(_entry("low_B", frozenset({0, 2, 3}), sig_seed=2))
    reg.add(_entry("low_C", frozenset({1, 2, 3}), sig_seed=3))
    # High-digit cluster
    reg.add(_entry("hi_A", frozenset({6, 7, 8}), sig_seed=4))
    reg.add(_entry("hi_B", frozenset({6, 8, 9}), sig_seed=5))
    reg.add(_entry("hi_C", frozenset({7, 8, 9}), sig_seed=6))

    res = recluster(reg, k=2, metric="jaccard")
    assert res.n_clusters == 2
    # Each cluster should be homogeneous — either all-low or all-high
    names_per_cluster = [set(c) for c in res.clusters]
    low_set = {"low_A", "low_B", "low_C"}
    hi_set = {"hi_A", "hi_B", "hi_C"}
    assert low_set in names_per_cluster
    assert hi_set in names_per_cluster


def test_recluster_anchors_are_unions():
    reg = FractalRegistry()
    reg.add(_entry("a", frozenset({0, 1}), sig_seed=1))
    reg.add(_entry("b", frozenset({1, 2}), sig_seed=2))
    reg.add(_entry("c", frozenset({7, 8}), sig_seed=3))
    reg.add(_entry("d", frozenset({8, 9}), sig_seed=4))

    res = recluster(reg, k=2, metric="jaccard")
    assert res.n_clusters == 2
    # Anchors should be unions of the member label sets
    anchors = sorted([sorted(a) for a in res.anchors])
    assert [0, 1, 2] in anchors
    assert [7, 8, 9] in anchors


def test_recluster_signature_metric_fallback():
    reg = FractalRegistry()
    # Two widely-separated signature blobs
    for i in range(3):
        rng = np.random.RandomState(i)
        reg.add(FractalEntry(
            name=f"a{i}",
            signature=(rng.randn(10) + np.array([5.0] * 10)).astype(np.float64),
            metadata={"task": "a"},
        ))
    for i in range(3):
        rng = np.random.RandomState(100 + i)
        reg.add(FractalEntry(
            name=f"b{i}",
            signature=(rng.randn(10) + np.array([-5.0] * 10)).astype(np.float64),
            metadata={"task": "b"},
        ))
    res = recluster(reg, k=2, metric="signature")
    assert res.n_clusters == 2
    names_per_cluster = [set(c) for c in res.clusters]
    assert {"a0", "a1", "a2"} in names_per_cluster
    assert {"b0", "b1", "b2"} in names_per_cluster


def test_recluster_deterministic():
    reg = FractalRegistry()
    for i, lbl in enumerate([{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}]):
        reg.add(_entry(f"e{i}", frozenset(lbl), sig_seed=i))
    r1 = recluster(reg, k=3, metric="jaccard")
    r2 = recluster(reg, k=3, metric="jaccard")
    # Determinism: same cluster membership
    assert [sorted(c) for c in r1.clusters] == [sorted(c) for c in r2.clusters]


def test_recluster_small_registry():
    reg = FractalRegistry()
    reg.add(_entry("only", frozenset({0}), sig_seed=1))
    res = recluster(reg, k=3, metric="jaccard")
    # Single-entry registry returns one cluster with that entry
    assert res.n_clusters == 1
    assert res.clusters == [["only"]]


def test_recluster_missing_task_labels_raises():
    reg = FractalRegistry()
    reg.add(FractalEntry(
        name="a", signature=np.zeros(10),
        metadata={"task": "a"},  # no task_labels
    ))
    reg.add(FractalEntry(
        name="b", signature=np.ones(10),
        metadata={"task": "b"},
    ))
    with pytest.raises(ValueError, match="missing metadata"):
        recluster(reg, k=2, metric="jaccard")


def test_recluster_result_to_dict():
    reg = FractalRegistry()
    reg.add(_entry("a", frozenset({0, 1}), sig_seed=1))
    reg.add(_entry("b", frozenset({2, 3}), sig_seed=2))
    res = recluster(reg, k=2, metric="jaccard")
    d = res.to_dict()
    assert "clusters" in d and "anchors" in d and "k" in d
    assert d["metric"] == "jaccard"
    assert d["k"] == 2
    assert d["n_clusters"] == 2
