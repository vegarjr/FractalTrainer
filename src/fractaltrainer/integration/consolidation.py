"""Direction G — cross-task distillation / consolidation.

The registry stores N independently-trained specialists. At inference,
routing picks one (match) or blends a few (compose) per query. At
large N this is operationally awkward: to answer one query we touch
1–K specialist models.

Consolidation trains a *generalist* — one model that mimics the
registry's softmax outputs on a broad probe set. At query time, the
generalist handles queries it's confident about; only low-confidence
queries fall through to the specialist router. This is analogous to
how human memory consolidates specific experiences into general
patterns, with specific memories remaining accessible when needed.

Mechanism:
    1. Collect a probe batch X_probe (typically the same one used
       for signatures).
    2. For each x in X_probe, compute the "teacher" output as the
       compose-blend of all registry specialists (inverse-distance-
       weighted by a query signature the generalist will itself
       approximate).
    3. Train the generalist via KL-divergence on the teacher's
       softmax. This is standard knowledge distillation (Hinton et
       al., 2015) with the twist that the teacher is an ensemble of
       classifier outputs rather than a single model.
    4. At query time: compute generalist prediction + confidence
       (e.g. 1 − entropy/log(n_classes)). If confidence ≥ threshold,
       use generalist. Else fall through to the specialist router
       (match/compose/spawn).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fractaltrainer.integration.context_mlp import ContextAwareMLP
from fractaltrainer.registry import FractalRegistry


@dataclass
class DistillStats:
    n_steps: int
    final_loss: float
    loss_history: list[float]
    elapsed_s: float


def teacher_output(
    models: Sequence[nn.Module],
    x: torch.Tensor,
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Aggregate specialists' softmax outputs into a teacher distribution.

    Default: uniform mean over all specialists. Pass explicit
    weights (e.g. inverse-distance softmax on a query signature) to
    approximate the registry's compose-verdict behavior.
    """
    if not models:
        raise ValueError("teacher_output requires ≥1 specialist model")
    outs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            try:
                logits = m(x, context=None)
            except TypeError:
                logits = m(x)
            outs.append(F.softmax(logits, dim=1))
    stacked = torch.stack(outs, dim=0)  # (M, B, K)
    if weights is None:
        return stacked.mean(dim=0)
    w = torch.tensor(list(weights), dtype=torch.float32).view(-1, 1, 1)
    w = w / w.sum()
    return (stacked * w).sum(dim=0)


def train_generalist(
    specialists: Sequence[nn.Module],
    probe_loader: Iterable,
    *,
    generalist_factory: Callable[[], nn.Module] | None = None,
    n_steps: int = 500,
    lr: float = 0.01,
    temperature: float = 2.0,
    seed: int = 42,
) -> tuple[nn.Module, DistillStats]:
    """Distill N specialists into one generalist via soft-label KD.

    The generalist is trained on `probe_loader` (which yields `(x, _)`
    — labels are ignored) with soft labels from the uniform-mean
    teacher over `specialists`. Temperature controls how much the
    teacher's confidence peaks: T=1 reproduces teacher softmax, T>1
    smooths it.
    """
    if generalist_factory is None:
        generalist_factory = lambda: ContextAwareMLP(context_scale=0.0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    generalist = generalist_factory()
    optimizer = torch.optim.Adam(generalist.parameters(), lr=lr)

    loss_history: list[float] = []
    t0 = time.time()
    step = 0
    it = iter(probe_loader)
    while step < n_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(probe_loader)
            batch = next(it)
        x = batch[0] if isinstance(batch, (tuple, list)) else batch

        with torch.no_grad():
            teacher = teacher_output(specialists, x)  # (B, K)

        # Generalist logits at temperature T
        try:
            logits = generalist(x, context=None)
        except TypeError:
            logits = generalist(x)
        student_log_probs = F.log_softmax(logits / temperature, dim=1)
        teacher_smoothed = F.softmax(
            torch.log(teacher.clamp(min=1e-9)) / temperature, dim=1,
        )
        loss = F.kl_div(student_log_probs, teacher_smoothed,
                         reduction="batchmean") * (temperature ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.item()))
        step += 1

    return generalist, DistillStats(
        n_steps=step,
        final_loss=loss_history[-1] if loss_history else float("nan"),
        loss_history=loss_history,
        elapsed_s=time.time() - t0,
    )


@dataclass
class ConsolidatedDecision:
    """Output of a consolidated-router lookup.

    Attributes:
        used_generalist: True if the generalist answered.
        prediction: the chosen class (argmax logit) from whichever
            path answered.
        confidence: 1 − normalized_entropy of the answering path.
        generalist_confidence: the generalist's confidence (independent
            of whether it actually answered).
        specialist_used: entry name if specialists answered, else None.
    """

    used_generalist: bool
    prediction: int
    confidence: float
    generalist_confidence: float
    specialist_used: str | None = None


class ConsolidatedRouter:
    """Generalist-first router with specialist fall-through.

    Usage:
        cr = ConsolidatedRouter(generalist, registry, models_by_entry,
                                 confidence_threshold=0.85)
        decision = cr.predict(x_single_sample, query_signature)

    For batched prediction, loop over samples or call `.predict_batch`.
    """

    def __init__(
        self,
        generalist: nn.Module,
        registry: FractalRegistry,
        model_by_entry: dict[str, nn.Module],
        *,
        confidence_threshold: float = 0.85,
        n_classes: int = 10,
    ):
        self.generalist = generalist
        self.registry = registry
        self.model_by_entry = model_by_entry
        self.confidence_threshold = float(confidence_threshold)
        self.n_classes = int(n_classes)
        self._log_n = float(np.log(max(2, n_classes)))

    def _confidence(self, probs: torch.Tensor) -> float:
        """1 − H(p) / log(n_classes); 1.0 = fully confident, 0.0 = uniform."""
        p = probs.clamp(min=1e-9)
        ent = -(p * torch.log(p)).sum(dim=-1).item()
        return 1.0 - ent / self._log_n

    def _generalist_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.generalist.eval()
        with torch.no_grad():
            try:
                logits = self.generalist(x, context=None)
            except TypeError:
                logits = self.generalist(x)
        return F.softmax(logits, dim=1)

    def predict(
        self,
        x: torch.Tensor,
        query_signature: np.ndarray | None = None,
    ) -> ConsolidatedDecision:
        """Answer a single query.

        If generalist confidence ≥ threshold → use it.
        Else route via signature: if query_signature provided, find
        nearest specialist and use its prediction. Without a
        signature, fall back to generalist anyway.
        """
        if x.ndim == 3:  # (C, H, W) → (1, C, H, W)
            x = x.unsqueeze(0)
        gen_probs = self._generalist_forward(x)  # (1, K)
        gen_conf = self._confidence(gen_probs[0])
        gen_pred = int(gen_probs[0].argmax().item())

        if gen_conf >= self.confidence_threshold:
            return ConsolidatedDecision(
                used_generalist=True,
                prediction=gen_pred,
                confidence=gen_conf,
                generalist_confidence=gen_conf,
            )

        if query_signature is None:
            # No routing info → best-effort use generalist even at low conf
            return ConsolidatedDecision(
                used_generalist=True,
                prediction=gen_pred,
                confidence=gen_conf,
                generalist_confidence=gen_conf,
            )

        retrieval = self.registry.find_nearest(query_signature, k=1)
        if not retrieval.entries:
            return ConsolidatedDecision(
                used_generalist=True,
                prediction=gen_pred, confidence=gen_conf,
                generalist_confidence=gen_conf,
            )
        nearest = retrieval.entries[0]
        model = self.model_by_entry.get(nearest.name)
        if model is None:
            return ConsolidatedDecision(
                used_generalist=True,
                prediction=gen_pred, confidence=gen_conf,
                generalist_confidence=gen_conf,
            )
        model.eval()
        with torch.no_grad():
            try:
                logits = model(x, context=None)
            except TypeError:
                logits = model(x)
            spec_probs = F.softmax(logits, dim=1)
        spec_conf = self._confidence(spec_probs[0])
        spec_pred = int(spec_probs[0].argmax().item())

        return ConsolidatedDecision(
            used_generalist=False,
            prediction=spec_pred,
            confidence=spec_conf,
            generalist_confidence=gen_conf,
            specialist_used=nearest.name,
        )

    def predict_batch(
        self,
        x: torch.Tensor,
        query_signature: np.ndarray | None = None,
    ) -> list[ConsolidatedDecision]:
        """Answer each sample in a batch independently."""
        return [self.predict(x[i], query_signature) for i in range(x.size(0))]
