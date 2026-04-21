"""Evaluation utilities — accuracy on held-out loader, sample-efficiency curves.

The demo's ablation table compares three spawn arms at five budgets
(N ∈ {50, 100, 300, 500, 1000}). This module produces those numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from fractaltrainer.integration.context_mlp import ContextAwareMLP


def evaluate_expert(
    model: torch.nn.Module,
    eval_loader: Iterable,
    *,
    context: torch.Tensor | None = None,
    device: str = "cpu",
) -> float:
    """Return accuracy (scalar in [0, 1]) on an eval dataloader.

    For ContextAwareMLP: context is always None at evaluation time
    (the probe-signature invariant — see Sprint 17 plan). Legacy models
    that ignore the `context` kwarg still work.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in eval_loader:
            x = x.to(device)
            y = y.to(device)
            try:
                logits = model(x, context=context)
            except TypeError:
                logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1) if total else 0.0


@dataclass
class BudgetResult:
    budget: int
    seed: int
    arm: str
    accuracy: float
    final_loss: float
    elapsed_s: float
    notes: dict = field(default_factory=dict)


@dataclass
class SampleEfficiencyResult:
    arm: str
    budgets: list[int]
    per_seed: dict[int, dict[int, float]]  # seed → budget → accuracy
    mean_by_budget: dict[int, float]
    stdev_by_budget: dict[int, float]
    raw: list[BudgetResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "budgets": list(self.budgets),
            "per_seed": {
                str(s): {str(b): float(a) for b, a in inner.items()}
                for s, inner in self.per_seed.items()
            },
            "mean_by_budget": {str(b): float(v) for b, v in self.mean_by_budget.items()},
            "stdev_by_budget": {str(b): float(v) for b, v in self.stdev_by_budget.items()},
            "raw": [r.__dict__ for r in self.raw],
        }


def sample_efficiency_curve(
    run_arm: Callable[[int, int], tuple[torch.nn.Module, float]],
    *,
    arm_name: str,
    budgets: Sequence[int],
    seeds: Sequence[int],
    eval_loader: Iterable,
) -> SampleEfficiencyResult:
    """Run one ablation arm across (budget × seed) grid; return aggregated stats.

    Args:
        run_arm: callable (budget, seed) -> (trained_model, final_loss).
            This is the arm-specific training hook — e.g. for arm B it's
            a closure that calls spawn_with_context with the given budget
            as n_steps and the given seed. Arm A / arm C are analogous.
        arm_name: label used in the result (for the table).
        budgets: list of training-step counts to test.
        seeds: list of seeds to average over.
        eval_loader: shared evaluation dataloader (same across arms for
            fair comparison).

    Returns:
        SampleEfficiencyResult with per-seed accuracies + mean/stdev.
    """
    per_seed: dict[int, dict[int, float]] = {s: {} for s in seeds}
    raw: list[BudgetResult] = []
    for b in budgets:
        for s in seeds:
            t0 = time.time()
            model, final_loss = run_arm(b, s)
            elapsed = time.time() - t0
            acc = evaluate_expert(model, eval_loader)
            per_seed[s][b] = acc
            raw.append(BudgetResult(
                budget=b, seed=s, arm=arm_name, accuracy=acc,
                final_loss=final_loss, elapsed_s=elapsed,
            ))

    mean_by_budget: dict[int, float] = {}
    stdev_by_budget: dict[int, float] = {}
    for b in budgets:
        accs = [per_seed[s].get(b, float("nan")) for s in seeds]
        accs = [a for a in accs if np.isfinite(a)]
        if accs:
            mean_by_budget[b] = float(np.mean(accs))
            stdev_by_budget[b] = float(np.std(accs, ddof=0))
        else:
            mean_by_budget[b] = float("nan")
            stdev_by_budget[b] = float("nan")

    return SampleEfficiencyResult(
        arm=arm_name, budgets=list(budgets),
        per_seed=per_seed,
        mean_by_budget=mean_by_budget,
        stdev_by_budget=stdev_by_budget,
        raw=raw,
    )


def render_efficiency_table_md(results: Sequence[SampleEfficiencyResult]) -> str:
    """Markdown table: rows=arms, cols=budgets, cells=mean ± stdev."""
    if not results:
        return ""
    budgets = list(results[0].budgets)
    header = "| Arm | " + " | ".join(f"N={b}" for b in budgets) + " |"
    sep = "|" + "|".join(["---"] * (len(budgets) + 1)) + "|"
    rows = []
    for r in results:
        cells = []
        for b in budgets:
            m = r.mean_by_budget.get(b, float("nan"))
            s = r.stdev_by_budget.get(b, float("nan"))
            cells.append(f"{m:.3f}±{s:.3f}")
        rows.append(f"| {r.arm} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)
