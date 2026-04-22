"""Direction D — self-spawning fractals.

The FractalPipeline's match/compose/spawn primitive handles an
individual query in isolation. Over time, the query stream carries
information the single-query router doesn't use: if many queries in
a row land in the compose band (same K neighbors blended), the
registry has a *gap* — a region where no single expert is a good fit
but the query load is real.

AutoSpawnPolicy observes the compose stream and, when a threshold of
compose verdicts accumulate in a similar signature region, proposes
a new expert. The new expert's task is defined as the *union* of the
K neighbors' class-1 label sets (so the proposal is a bridging task
that covers what the existing experts together cover, but as a single
specialist). The caller decides whether to execute the proposal —
the policy is informational.

This does not require external supervision: the decision to grow is
driven by the query stream alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from fractaltrainer.registry import FractalEntry, FractalRegistry, GrowthDecision


@dataclass
class SpawnProposal:
    """A policy-generated proposal to train a new expert.

    Attributes:
        centroid_signature: mean of observed compose-verdict query
            signatures. Acts as the target region for the new expert.
        neighbor_entries: K entries nearest to the centroid, sorted
            by distance ascending.
        proposed_task_labels: union of neighbor_entries' class-1 label
            sets — a natural bridging task.
        n_queries_in_region: how many compose verdicts the policy
            observed before triggering.
        mean_distance_to_centroid: avg L2 of observed signatures to
            the proposed centroid (smaller = tighter cluster).
        max_distance_to_centroid: maximum L2; useful for deciding
            whether the cluster is meaningful or just many unrelated
            compose events.
    """

    centroid_signature: np.ndarray
    neighbor_entries: list[FractalEntry]
    proposed_task_labels: frozenset[int]
    n_queries_in_region: int
    mean_distance_to_centroid: float
    max_distance_to_centroid: float

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries_in_region,
            "neighbors": [e.name for e in self.neighbor_entries],
            "proposed_task_labels": sorted(self.proposed_task_labels),
            "mean_distance_to_centroid": float(self.mean_distance_to_centroid),
            "max_distance_to_centroid": float(self.max_distance_to_centroid),
        }


class AutoSpawnPolicy:
    """Accumulate compose-verdict signatures; propose spawn on threshold.

    Usage:
        policy = AutoSpawnPolicy(trigger_threshold=5)
        for q in query_stream:
            decision = pipeline.step(q).decision
            if decision.verdict == "compose":
                policy.observe_compose(q_signature)
            if policy.should_propose():
                proposal = policy.propose(registry)
                # caller decides to execute or ignore
                policy.reset()
    """

    def __init__(
        self,
        trigger_threshold: int = 5,
        k_neighbors: int = 3,
        max_cluster_radius: float | None = None,
        suppress_redundant: bool = True,
    ):
        """
        Args:
            trigger_threshold: how many compose verdicts accumulate
                before a proposal is generated.
            k_neighbors: how many nearest entries to include in the
                proposal (and thus the label-set union).
            max_cluster_radius: optional cap on mean distance from
                centroid. If the observed compose cluster is spread
                wider than this, the proposal is suppressed (the
                compose verdicts are not about *one* gap, they're
                scattered).
            suppress_redundant: if True (default), suppress proposals
                whose proposed_task_labels is a subset of the nearest
                neighbor's task_labels. This avoids duplicating an
                existing task when the compose cluster lands on a
                single well-represented task rather than bridging
                multiple (Review 39 fix).
        """
        self.trigger_threshold = int(trigger_threshold)
        self.k_neighbors = int(k_neighbors)
        self.max_cluster_radius = max_cluster_radius
        self.suppress_redundant = bool(suppress_redundant)
        self._compose_signatures: list[np.ndarray] = []
        self._proposals_history: list[SpawnProposal] = []
        self._suppressed_redundant_count = 0

    def observe_compose(self, query_signature: np.ndarray) -> None:
        """Record that one compose-verdict query with this signature
        just occurred."""
        if not isinstance(query_signature, np.ndarray):
            query_signature = np.asarray(query_signature)
        self._compose_signatures.append(query_signature.copy())

    def observe(self, query_signature: np.ndarray,
                 decision: GrowthDecision) -> None:
        """Convenience helper — only records if verdict is compose."""
        if decision.verdict == "compose":
            self.observe_compose(query_signature)

    @property
    def compose_count(self) -> int:
        return len(self._compose_signatures)

    def should_propose(self) -> bool:
        return self.compose_count >= self.trigger_threshold

    def propose(self, registry: FractalRegistry,
                 task_label_key: str = "task_labels",
                 ) -> SpawnProposal | None:
        """Build a SpawnProposal from the accumulated compose signatures.

        Returns None if the threshold hasn't been reached or if the
        observed cluster is too wide (max_cluster_radius check).
        """
        if not self.should_propose():
            return None
        if len(registry) == 0:
            return None

        sigs = np.stack(self._compose_signatures, axis=0)
        centroid = sigs.mean(axis=0)

        dists_to_centroid = np.linalg.norm(sigs - centroid, axis=1)
        mean_d = float(dists_to_centroid.mean())
        max_d = float(dists_to_centroid.max())

        if self.max_cluster_radius is not None and mean_d > self.max_cluster_radius:
            return None  # too scattered to be about one gap

        retrieval = registry.find_nearest(centroid, k=self.k_neighbors)
        union_labels: set[int] = set()
        for e in retrieval.entries:
            labels = e.metadata.get(task_label_key)
            if labels is not None:
                union_labels |= set(int(x) for x in labels)

        # Suppress if the union is a subset of the nearest neighbor's
        # labels — means the compose cluster landed on a single
        # already-covered task, not between tasks (Review 39 fix).
        if self.suppress_redundant and retrieval.entries:
            nearest_labels = set(int(x) for x in (
                retrieval.entries[0].metadata.get(task_label_key) or []
            ))
            if nearest_labels and union_labels <= nearest_labels:
                self._suppressed_redundant_count += 1
                return None

        proposal = SpawnProposal(
            centroid_signature=centroid,
            neighbor_entries=list(retrieval.entries),
            proposed_task_labels=frozenset(union_labels),
            n_queries_in_region=self.compose_count,
            mean_distance_to_centroid=mean_d,
            max_distance_to_centroid=max_d,
        )
        self._proposals_history.append(proposal)
        return proposal

    @property
    def suppressed_redundant_count(self) -> int:
        return self._suppressed_redundant_count

    def reset(self) -> None:
        """Clear the accumulated compose signatures. Call after a
        proposal has been executed (or explicitly rejected)."""
        self._compose_signatures = []

    @property
    def history(self) -> list[SpawnProposal]:
        """All proposals emitted so far (in order)."""
        return list(self._proposals_history)
