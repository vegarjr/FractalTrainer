"""GeometricRepairContext — what the LLM sees when shape diverges.

Replaces Fractal's test-failure / routing-confusion context with a
geometry-first bundle: trajectory summary, target, divergence, loss trend,
prior attempts, and the (single) patchable config file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fractaltrainer.repair.hparam_config import load_hparams
from fractaltrainer.target.target_shape import TargetShape


ALLOWED_FILES = ("configs/hparams.yaml",)


@dataclass
class GeometricRepairContext:
    current_hparams: dict
    hparams_yaml: str
    trajectory_summary: dict
    target: TargetShape
    divergence_score: float
    current_dim: float
    loss_history_summary: dict
    previous_attempts: list[dict] = field(default_factory=list)
    allowed_files: list[str] = field(
        default_factory=lambda: list(ALLOWED_FILES))
    protected_files: list[str] = field(default_factory=list)
    max_lines_changed: int = 15


class GeometricContextGatherer:
    """Assembles a GeometricRepairContext from a comparison report + log."""

    def __init__(self, project_root: str):
        self.root = Path(project_root)

    def gather(
        self,
        comparison_report: dict,
        target: TargetShape,
        log_path: str | None = None,
        hparams_path: str | None = None,
    ) -> GeometricRepairContext:
        """Build context for the repair prompt.

        Args:
            comparison_report: dict produced by scripts/run_comparison.py.
            target: the loaded target shape.
            log_path: optional path to results/repair_log.jsonl — prior attempts.
            hparams_path: optional override for configs/hparams.yaml.
        """
        hparams_rel = hparams_path or "configs/hparams.yaml"
        hparams_abs = self.root / hparams_rel
        current_hparams = load_hparams(hparams_abs)
        hparams_yaml = hparams_abs.read_text()

        trajectory_summary = {
            "shape": comparison_report.get("trajectory_shape"),
            "projected_shape": comparison_report.get("projection", {}).get(
                "projected_shape"),
            "primary": comparison_report.get("primary_result", {}),
            "sanity_box_counting_3d": comparison_report.get(
                "sanity_box_counting_3d", {}),
            "baseline_metrics": comparison_report.get("baseline_metrics", {}),
            # Only present when target.method == "golden_run_match":
            "golden_run": comparison_report.get("golden_run"),
        }

        divergence = comparison_report.get("divergence", {})
        primary = comparison_report.get("primary_result", {})

        loss_history_summary = self._summarize_loss_history(
            comparison_report.get("trajectory_path"))

        previous_attempts = self._load_previous_attempts(log_path)

        return GeometricRepairContext(
            current_hparams=current_hparams,
            hparams_yaml=hparams_yaml,
            trajectory_summary=trajectory_summary,
            target=target,
            divergence_score=float(divergence.get("score", float("nan"))),
            current_dim=float(primary.get("dim", float("nan"))),
            loss_history_summary=loss_history_summary,
            previous_attempts=previous_attempts,
            allowed_files=[hparams_rel],
            protected_files=[],
            max_lines_changed=15,
        )

    def _summarize_loss_history(self, trajectory_path: str | None) -> dict:
        """Pull loss_history from the meta JSON sibling of the trajectory npy."""
        if not trajectory_path:
            return {}
        traj_path = Path(trajectory_path)
        meta_path = traj_path.parent / (
            traj_path.stem.removesuffix("_trajectory") + "_meta.json")
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            return {}
        losses = meta.get("loss_history") or []
        if not losses:
            return {"n_steps": 0}
        losses_nan_free = [x for x in losses if isinstance(x, (int, float))]
        if not losses_nan_free:
            return {"n_steps": len(losses)}
        n = len(losses_nan_free)
        summary = {
            "n_steps": n,
            "first_loss": float(losses_nan_free[0]),
            "last_loss": float(losses_nan_free[-1]),
            "min_loss": float(min(losses_nan_free)),
            "max_loss": float(max(losses_nan_free)),
            "trend": "decreasing" if losses_nan_free[-1] < losses_nan_free[0]
                    else "increasing_or_flat",
        }
        # final-third mean vs first-third mean
        if n >= 6:
            third = n // 3
            summary["first_third_mean"] = float(
                sum(losses_nan_free[:third]) / third)
            summary["last_third_mean"] = float(
                sum(losses_nan_free[-third:]) / third)
        return summary

    def _load_previous_attempts(self, log_path: str | None,
                                max_attempts: int = 3) -> list[dict]:
        if not log_path:
            return []
        p = Path(log_path)
        if not p.exists():
            return []
        attempts: list[dict] = []
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        attempts.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []
        return attempts[-max_attempts:]
