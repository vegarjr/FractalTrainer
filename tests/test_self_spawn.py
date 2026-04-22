"""Unit tests for AutoSpawnPolicy (Direction D self-spawning)."""

from __future__ import annotations

import numpy as np
import pytest

from fractaltrainer.integration.self_spawn import (
    AutoSpawnPolicy,
    SpawnProposal,
)
from fractaltrainer.registry import FractalEntry, FractalRegistry


def _entry(name: str, labels, sig_offset: float) -> FractalEntry:
    """Make a FractalEntry with a one-hot-ish signature at fixed L2 from origin."""
    sig = np.zeros(10, dtype=np.float64)
    sig[0] = float(sig_offset)
    return FractalEntry(
        name=name, signature=sig,
        metadata={"task_labels": list(sorted(labels))},
    )


def _sig(offset: float) -> np.ndarray:
    s = np.zeros(10, dtype=np.float64)
    s[0] = float(offset)
    return s


def test_threshold_not_met_yields_none():
    reg = FractalRegistry()
    reg.add(_entry("a", [0, 1], 0.0))
    reg.add(_entry("b", [2, 3], 1.0))
    policy = AutoSpawnPolicy(trigger_threshold=5)
    for _ in range(3):
        policy.observe_compose(_sig(0.5))
    assert policy.propose(reg) is None
    assert not policy.should_propose()


def test_threshold_met_yields_proposal():
    reg = FractalRegistry()
    reg.add(_entry("a", [0, 1], 0.0))
    reg.add(_entry("b", [2, 3], 1.0))
    reg.add(_entry("c", [8, 9], 10.0))
    policy = AutoSpawnPolicy(trigger_threshold=5)
    for _ in range(5):
        policy.observe_compose(_sig(0.5))
    assert policy.should_propose()
    proposal = policy.propose(reg)
    assert proposal is not None
    assert proposal.n_queries_in_region == 5
    # centroid should be close to 0.5 on the first dim
    assert abs(proposal.centroid_signature[0] - 0.5) < 1e-6
    # nearest 3 by L2 to centroid: a (d=0.5), b (d=0.5), c (d=9.5)
    names = {e.name for e in proposal.neighbor_entries}
    assert "a" in names and "b" in names


def test_proposal_union_of_labels():
    reg = FractalRegistry()
    reg.add(_entry("low",  [0, 1, 2], 0.0))
    reg.add(_entry("mid",  [3, 4, 5], 0.1))
    reg.add(_entry("high", [6, 7, 8], 0.2))
    policy = AutoSpawnPolicy(trigger_threshold=3, k_neighbors=3)
    for _ in range(3):
        policy.observe_compose(_sig(0.1))
    proposal = policy.propose(reg)
    assert proposal is not None
    # Union of all three neighbors' label sets
    assert proposal.proposed_task_labels == frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8})


def test_scattered_cluster_rejected_by_radius():
    reg = FractalRegistry()
    reg.add(_entry("a", [0, 1], 0.0))
    policy = AutoSpawnPolicy(trigger_threshold=4, max_cluster_radius=0.5)
    # Widely-scattered compose signatures (mean distance > 0.5)
    policy.observe_compose(_sig(0.0))
    policy.observe_compose(_sig(2.0))
    policy.observe_compose(_sig(5.0))
    policy.observe_compose(_sig(10.0))
    proposal = policy.propose(reg)
    assert proposal is None  # too scattered


def test_tight_cluster_accepted():
    reg = FractalRegistry()
    reg.add(_entry("a", [0, 1], 0.0))
    reg.add(_entry("b", [4, 5], 1.0))  # add second distinct-task entry
    policy = AutoSpawnPolicy(trigger_threshold=3, max_cluster_radius=0.5)
    # Tightly clustered compose signatures
    policy.observe_compose(_sig(0.4))
    policy.observe_compose(_sig(0.5))
    policy.observe_compose(_sig(0.6))
    proposal = policy.propose(reg)
    assert proposal is not None


def test_reset_clears_history_but_keeps_proposals():
    reg = FractalRegistry()
    reg.add(_entry("a", [0, 1], 0.0))
    reg.add(_entry("b", [4, 5], 1.0))  # distinct task so proposal isn't redundant
    policy = AutoSpawnPolicy(trigger_threshold=2)
    policy.observe_compose(_sig(0.3))
    policy.observe_compose(_sig(0.3))
    p1 = policy.propose(reg)
    assert p1 is not None
    policy.reset()
    assert policy.compose_count == 0
    # History of proposals persists
    assert len(policy.history) == 1
    assert policy.history[0] is p1


def test_observe_delegates_for_compose_verdicts_only():
    from fractaltrainer.registry import GrowthDecision, RetrievalResult
    policy = AutoSpawnPolicy(trigger_threshold=3)

    match_decision = GrowthDecision(
        verdict="match", min_distance=0.1,
        retrieval=RetrievalResult(query_name="q", entries=[], distances=[]),
    )
    compose_decision = GrowthDecision(
        verdict="compose", min_distance=5.0,
        retrieval=RetrievalResult(query_name="q", entries=[], distances=[]),
    )
    spawn_decision = GrowthDecision(
        verdict="spawn", min_distance=10.0,
        retrieval=RetrievalResult(query_name="q", entries=[], distances=[]),
    )

    s = _sig(0.5)
    policy.observe(s, match_decision)    # ignored
    policy.observe(s, compose_decision)  # recorded
    policy.observe(s, spawn_decision)    # ignored
    assert policy.compose_count == 1


def test_empty_registry_yields_none():
    reg = FractalRegistry()  # empty
    policy = AutoSpawnPolicy(trigger_threshold=2)
    policy.observe_compose(_sig(0.5))
    policy.observe_compose(_sig(0.5))
    # Threshold met but registry empty
    assert policy.propose(reg) is None


def test_suppresses_redundant_proposal_by_default():
    """Union ⊆ nearest.task_labels → proposal suppressed."""
    reg = FractalRegistry()
    # Three entries of the SAME task — K=3 nearest will all be the
    # same task, union = that task, so proposal is redundant.
    reg.add(_entry("subset_A_seed1", [0, 1, 2, 3, 4], 0.0))
    reg.add(_entry("subset_A_seed2", [0, 1, 2, 3, 4], 0.1))
    reg.add(_entry("subset_A_seed3", [0, 1, 2, 3, 4], 0.2))
    policy = AutoSpawnPolicy(trigger_threshold=3)  # default suppress_redundant=True
    for _ in range(3):
        policy.observe_compose(_sig(0.05))
    proposal = policy.propose(reg)
    assert proposal is None
    assert policy.suppressed_redundant_count == 1


def test_nonredundant_proposal_passes():
    """When K=3 nearest span multiple tasks, union is novel → proposal fires."""
    reg = FractalRegistry()
    reg.add(_entry("low",  [0, 1, 2], 0.0))
    reg.add(_entry("mid",  [3, 4, 5], 0.1))
    reg.add(_entry("high", [6, 7, 8], 0.2))
    policy = AutoSpawnPolicy(trigger_threshold=3)  # default suppress_redundant=True
    for _ in range(3):
        policy.observe_compose(_sig(0.1))
    proposal = policy.propose(reg)
    assert proposal is not None
    # Union {0..8} is NOT a subset of any single neighbor's labels
    assert proposal.proposed_task_labels == frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8})


def test_suppress_off_allows_redundant():
    """suppress_redundant=False lets redundant proposals through."""
    reg = FractalRegistry()
    reg.add(_entry("subset_A_seed1", [0, 1, 2, 3, 4], 0.0))
    reg.add(_entry("subset_A_seed2", [0, 1, 2, 3, 4], 0.1))
    policy = AutoSpawnPolicy(trigger_threshold=2, suppress_redundant=False)
    policy.observe_compose(_sig(0.05))
    policy.observe_compose(_sig(0.05))
    proposal = policy.propose(reg)
    assert proposal is not None
    assert proposal.proposed_task_labels == frozenset({0, 1, 2, 3, 4})


def test_proposal_to_dict_is_json_safe():
    reg = FractalRegistry()
    reg.add(_entry("a", [0, 1], 0.0))
    reg.add(_entry("b", [2, 3], 0.1))
    policy = AutoSpawnPolicy(trigger_threshold=2)
    policy.observe_compose(_sig(0.05))
    policy.observe_compose(_sig(0.05))
    proposal = policy.propose(reg)
    d = proposal.to_dict()
    import json
    # Round-trip through JSON
    s = json.dumps(d)
    assert json.loads(s) == d
