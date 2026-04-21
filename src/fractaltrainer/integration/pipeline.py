"""FractalPipeline — closed perception→growth orchestrator.

Chains the existing primitives into one pass per incoming query:

    1. DESCRIBER  — turn raw (digit, label) pairs into a frozenset
                    of class-1 digits (Describer or MockDescriber).
    2. ROUTE      — signature the query via the caller-supplied
                    probe-signature callable; ask the FractalRegistry
                    for a GrowthDecision (match / compose / spawn).
    3. ACT        — dispatch based on verdict:
                      match   → return nearest expert's predictions
                      compose → coverage-weighted ensemble on top-K
                      spawn   → train a fresh context-aware expert on
                                this query's data, register it
    4. RECLUSTER  — every N spawns + at end of run, rebuild cluster
                    anchors from the registry's task_labels metadata.

The pipeline is a thin coordination layer; heavy lifting lives in the
primitives. Unit-testable via MockDescriber + synthetic signatures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import torch

from fractaltrainer.integration.context_injection import ContextSpec
from fractaltrainer.integration.context_mlp import ContextAwareMLP
from fractaltrainer.integration.describer_adapter import (
    DescribeResult,
    MockDescriber,
    OracleDescriber,
)
from fractaltrainer.integration.recluster import ClusteringResult, recluster
from fractaltrainer.integration.spawn import (
    TrainStats,
    spawn_baseline,
    spawn_random_context,
    spawn_with_context,
)
from fractaltrainer.registry import FractalEntry, FractalRegistry, GrowthDecision


@dataclass
class ReclusterPolicy:
    interval_spawns: int = 2
    trigger_at_end: bool = True


@dataclass
class PipelineStep:
    """Result of one pipeline.step call — enough context to render the demo table.

    Attributes:
        query_name: identifier the caller supplied.
        verdict: "match" / "compose" / "spawn" / "empty"
        describer: DescribeResult from step 1
        decision: GrowthDecision from the router
        action: which action fired — same as verdict unless spawn failed
        accuracy: evaluated accuracy of the chosen action on the caller's
            eval data; None if no accuracy was requested
        neighbors_used: names of the K nearest neighbors used for spawn
            context, empty list for match/compose
        train_stats: TrainStats if verdict was spawn, else None
        new_entry: FractalEntry created by a spawn, else None
        elapsed_s: wall clock for the step
    """

    query_name: str
    verdict: str
    describer: DescribeResult
    decision: GrowthDecision | None
    action: str
    accuracy: Optional[float] = None
    neighbors_used: list[str] = field(default_factory=list)
    train_stats: Optional[TrainStats] = None
    new_entry: Optional[FractalEntry] = None
    elapsed_s: float = 0.0
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query_name": self.query_name,
            "verdict": self.verdict,
            "action": self.action,
            "accuracy": self.accuracy,
            "neighbors_used": list(self.neighbors_used),
            "elapsed_s": float(self.elapsed_s),
            "describer": {
                "guess": sorted(self.describer.guess) if self.describer.guess else None,
                "positive_digits_seen": self.describer.positive_digits_seen,
                "negative_digits_seen": self.describer.negative_digits_seen,
                "raw_response_preview": self.describer.raw_response[:200],
            },
            "decision": self.decision.to_dict() if self.decision else None,
            "train_stats": (None if self.train_stats is None else {
                "n_steps": self.train_stats.n_steps,
                "final_loss": self.train_stats.final_loss,
                "elapsed_s": self.train_stats.elapsed_s,
            }),
            "new_entry_name": self.new_entry.name if self.new_entry else None,
            "notes": self.notes,
        }


@dataclass
class QueryInput:
    """Everything the pipeline needs for one query.

    The pipeline is intentionally decoupled from data loading — the
    caller provides pre-built loaders and a probe batch. This keeps
    pipeline.py testable with synthetic tensors and lets the demo
    script own the MNIST / torchvision code.

    Attributes:
        name: short identifier, e.g. "Q_spawn".
        pairs: (digit, binary_label) examples shown to the describer.
        truth_labels: ground-truth class-1 label set (for logging and
            for OracleDescriber).
        train_loader: dataloader for spawn training (ignored for
            match/compose).
        eval_loader: dataloader for accuracy measurement.
        compute_signature: callable (training_loader) -> np.ndarray of
            shape matching registry signatures. Used for routing. The
            caller typically trains a 1-step probe model and signatures
            it, or bypasses to an existing entry's signature for testing.
    """

    name: str
    pairs: list[tuple[int, int]]
    truth_labels: frozenset[int]
    train_loader: Any = None
    eval_loader: Any = None
    compute_signature: Optional[Callable[[], np.ndarray]] = None
    probe: Optional[torch.Tensor] = None


class FractalPipeline:
    """Closed perception→growth pipeline."""

    def __init__(
        self,
        registry: FractalRegistry,
        describer: Any,
        *,
        match_threshold: float = 5.0,
        spawn_threshold: float = 7.0,
        compose_k: int = 3,
        context_spec: ContextSpec | None = None,
        context_scale: float = 1.0,
        recluster_policy: ReclusterPolicy | None = None,
        task_label_key: str = "task_labels",
        model_by_entry: Optional[dict[str, torch.nn.Module]] = None,
    ):
        self.registry = registry
        self.describer = describer
        self.match_threshold = float(match_threshold)
        self.spawn_threshold = float(spawn_threshold)
        self.compose_k = int(compose_k)
        self.context_spec = context_spec or ContextSpec(k=compose_k)
        self.context_scale = float(context_scale)
        self.recluster_policy = recluster_policy or ReclusterPolicy()
        self.task_label_key = task_label_key
        # Caller-owned map from entry name → trained model, used when
        # the verdict is match/compose (registry itself only stores
        # signatures). If a spawn creates a new model, the pipeline
        # adds it here.
        self.model_by_entry: dict[str, torch.nn.Module] = (
            dict(model_by_entry) if model_by_entry else {}
        )
        self.spawn_counter = 0
        self.cluster_history: list[ClusteringResult] = []

    def register_model(self, entry_name: str, model: torch.nn.Module) -> None:
        """Associate a trained model with an existing FractalEntry."""
        self.model_by_entry[entry_name] = model

    def _describe(self, q: QueryInput) -> DescribeResult:
        if isinstance(self.describer, OracleDescriber):
            self.describer.set_truth(q.truth_labels)
        return self.describer.describe(q.pairs)

    def _decide(self, signature: np.ndarray) -> GrowthDecision:
        return self.registry.decide(
            signature,
            match_threshold=self.match_threshold,
            spawn_threshold=self.spawn_threshold,
            compose_k=self.compose_k,
        )

    def _neighbor_models(
        self, decision: GrowthDecision,
    ) -> tuple[list[torch.nn.Module], list[str], list[float]]:
        """Extract the K nearest models + their distances from a decision."""
        if decision.retrieval is None:
            return [], [], []
        models: list[torch.nn.Module] = []
        names: list[str] = []
        dists: list[float] = []
        for entry, dist in zip(decision.retrieval.entries, decision.retrieval.distances):
            m = self.model_by_entry.get(entry.name)
            if m is None:
                continue
            models.append(m)
            names.append(entry.name)
            dists.append(float(dist))
        return models, names, dists

    def _maybe_recluster(self) -> Optional[ClusteringResult]:
        """Trigger reclustering per policy. Returns result if fired, else None."""
        if self.spawn_counter == 0:
            return None
        if self.spawn_counter % self.recluster_policy.interval_spawns != 0:
            return None
        try:
            r = recluster(self.registry, k=max(2, min(3, len(self.registry))),
                          metric="jaccard", task_label_key=self.task_label_key)
            self.cluster_history.append(r)
            return r
        except ValueError:
            return None

    def finalize(self) -> Optional[ClusteringResult]:
        """Call at end of a run to force a final reclustering."""
        if not self.recluster_policy.trigger_at_end:
            return None
        try:
            r = recluster(
                self.registry,
                k=max(2, min(3, len(self.registry))),
                metric="jaccard",
                task_label_key=self.task_label_key,
            )
            self.cluster_history.append(r)
            return r
        except ValueError:
            return None

    def step(
        self,
        q: QueryInput,
        *,
        signature: np.ndarray | None = None,
        spawn_mode: str = "neighbors",
        spawn_n_steps: int = 500,
        spawn_lr: float = 0.01,
        spawn_seed: int = 42,
        eval_fn: Optional[Callable[[torch.nn.Module, Any], float]] = None,
    ) -> PipelineStep:
        """Run one pipeline step.

        Args:
            q: QueryInput bundle.
            signature: pre-computed query signature. If None, caller
                must supply q.compute_signature.
            spawn_mode: "neighbors" (arm B), "none" (arm A),
                "random" (arm C). Only relevant when verdict=spawn.
            eval_fn: optional (model, eval_loader) -> float accuracy
                used to score the chosen action. Called on:
                    - match:   nearest expert's model
                    - compose: NOT scored here — compose ensembling is
                               a separate pathway the demo handles
                    - spawn:   the newly-trained model
        """
        t0 = time.time()
        describe_res = self._describe(q)

        # Routing signature
        if signature is None:
            if q.compute_signature is None:
                raise ValueError(
                    f"QueryInput {q.name!r} needs either a precomputed "
                    "signature or compute_signature callable")
            signature = q.compute_signature()

        decision = self._decide(signature)

        neighbors_used: list[str] = []
        train_stats: Optional[TrainStats] = None
        new_entry: Optional[FractalEntry] = None
        accuracy: Optional[float] = None
        action = decision.verdict
        notes: dict = {}

        if decision.verdict == "match":
            if eval_fn and decision.retrieval and decision.retrieval.entries:
                nearest = decision.retrieval.entries[0]
                model = self.model_by_entry.get(nearest.name)
                if model is not None and q.eval_loader is not None:
                    accuracy = eval_fn(model, q.eval_loader)
                    neighbors_used = [nearest.name]

        elif decision.verdict == "compose":
            # Compose ensembling is left to the caller because it
            # requires per-sample probabilities — the pipeline only
            # decides. We report the composite entry names so the
            # demo can run coverage-greedy on them.
            if decision.composite_entries:
                neighbors_used = [e.name for e in decision.composite_entries]
            notes["compose_weights"] = (
                [float(w) for w in decision.composite_weights]
                if decision.composite_weights is not None else None
            )

        elif decision.verdict == "spawn":
            if q.train_loader is None or q.probe is None:
                raise ValueError(
                    f"spawn verdict on query {q.name!r} requires "
                    "train_loader and probe on QueryInput")
            n_models, n_names, n_dists = self._neighbor_models(decision)
            neighbors_used = n_names
            entry_name = f"spawn_{q.name}_seed{spawn_seed}"

            if spawn_mode == "none" or not n_models:
                if spawn_mode == "neighbors" and not n_models:
                    notes["spawn_fallback"] = (
                        "requested neighbors but model_by_entry lacked "
                        "them — used baseline arm"
                    )
                model, new_entry, train_stats = spawn_baseline(
                    q.train_loader, q.probe,
                    n_steps=spawn_n_steps, lr=spawn_lr, seed=spawn_seed,
                    entry_name=entry_name, task=q.name,
                    metadata_extra={
                        "task_labels": list(sorted(q.truth_labels)),
                    },
                )
            elif spawn_mode == "neighbors":
                model, new_entry, train_stats = spawn_with_context(
                    q.train_loader, q.probe,
                    neighbors=n_models,
                    neighbor_distances=n_dists,
                    spec=self.context_spec,
                    context_scale=self.context_scale,
                    n_steps=spawn_n_steps, lr=spawn_lr, seed=spawn_seed,
                    entry_name=entry_name, task=q.name,
                    metadata_extra={
                        "task_labels": list(sorted(q.truth_labels)),
                    },
                )
            elif spawn_mode == "random":
                model, new_entry, train_stats = spawn_random_context(
                    q.train_loader, q.probe,
                    context_scale=self.context_scale,
                    n_steps=spawn_n_steps, lr=spawn_lr, seed=spawn_seed,
                    entry_name=entry_name, task=q.name,
                    metadata_extra={
                        "task_labels": list(sorted(q.truth_labels)),
                    },
                )
            else:
                raise ValueError(f"unknown spawn_mode: {spawn_mode!r}")

            self.registry.add(new_entry)
            self.model_by_entry[new_entry.name] = model
            self.spawn_counter += 1

            if eval_fn and q.eval_loader is not None:
                accuracy = eval_fn(model, q.eval_loader)

            rc = self._maybe_recluster()
            if rc is not None:
                notes["recluster_fired"] = {
                    "n_clusters": rc.n_clusters,
                    "anchors": [sorted(a) for a in rc.anchors],
                }

        elapsed = time.time() - t0
        return PipelineStep(
            query_name=q.name,
            verdict=decision.verdict,
            describer=describe_res,
            decision=decision,
            action=action,
            accuracy=accuracy,
            neighbors_used=neighbors_used,
            train_stats=train_stats,
            new_entry=new_entry,
            elapsed_s=elapsed,
            notes=notes,
        )
