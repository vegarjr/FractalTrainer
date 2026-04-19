"""ADAPTED from /home/vegar/Documents/Fractal/evolution/stage11_introspection/prompt_builder.py
Source commit: a910a9202c9d5e8b9f5f3c2c1c9e7d6b5a4b3c2a
Original author: Vegar Ratdal (Fractal project)
License: MIT (same ownership)
Modifications:
  - Rewrote SYSTEM_PROMPT for the geometric / hyperparameter-tuning task.
  - Rewrote build() to inject trajectory + target + divergence + loss-trend
    + prior-attempt history instead of failing-tests + source-code.
  - Constraints section now names configs/hparams.yaml and the six allowed
    keys explicitly.
"""

from __future__ import annotations

import json

from fractaltrainer.repair.context import GeometricRepairContext


class PromptBuilder:
    """Builds the user prompt for an LLM-driven hyperparameter patch."""

    SYSTEM_PROMPT = """You adjust training hyperparameters to move a PyTorch
training trajectory's fractal dimension toward a specified target.

You receive:
  - the current hparams (YAML)
  - a geometric summary of the most recent training run's weight trajectory
  - a target fractal dimension and tolerance band
  - the divergence score (|current_dim - target| / tolerance)
  - the loss trend
  - prior repair attempts (if any) with their dim deltas

You return ONE patch to configs/hparams.yaml, in the exact format below.
The patch must be a minimal, targeted change to ONE or a SMALL NUMBER of
hparam lines; never rewrite the whole file.

RULES:
1. Output ONLY the patch block. No prose before or after.
2. The ONLY file you may patch is `configs/hparams.yaml`.
3. Allowed hparam keys: learning_rate, batch_size, weight_decay, dropout,
   init_seed, optimizer. No other keys.
4. Ranges:
     learning_rate in [1e-6, 1.0]
     batch_size   in [1, 1024]
     weight_decay in [0.0, 0.1]
     dropout      in [0.0, 0.9]
     init_seed    in [0, 2^31 - 1]
     optimizer    in {{sgd, adam, adamw}}
5. Maximum {max_lines} total lines changed. Keep YAML layout + comments
   intact as much as possible.
6. If you believe no hparam change will help, return exactly:
     NO_FIX_FOUND: <short reason>

PATCH FORMAT (verbatim, including the triple-angle markers):
<<<PATCH
file: configs/hparams.yaml
---old---
<exact text to find in configs/hparams.yaml, must match verbatim and be unique>
---new---
<replacement text>
>>>PATCH"""

    def build(self, context: GeometricRepairContext) -> str:
        parts: list[str] = []

        # -- Target + divergence --
        t = context.target
        parts.append("## TARGET SHAPE\n")
        parts.append(f"- method: {t.method}")
        parts.append(f"- dim_target: {t.dim_target}")
        parts.append(f"- tolerance: {t.tolerance}  "
                     f"(accept band: [{t.band_low:.3f}, {t.band_high:.3f}])")
        parts.append(f"- projection: {t.projection.method} -> "
                     f"{t.projection.n_components} components "
                     f"(seed={t.projection.seed})")

        parts.append("\n## CURRENT MEASUREMENT\n")
        parts.append(f"- current_dim: {context.current_dim:.4f}")
        parts.append(f"- divergence_score: {context.divergence_score:.3f} "
                     "(0=center of band, 1=at band edge, >1=outside)")
        parts.append(f"- trajectory shape: {context.trajectory_summary.get('shape')}")
        parts.append(f"- projected shape: "
                     f"{context.trajectory_summary.get('projected_shape')}")

        primary = context.trajectory_summary.get("primary", {})
        if primary:
            parts.append("\n### primary_result details\n```json")
            parts.append(json.dumps(primary, indent=2, default=str))
            parts.append("```")

        baseline = context.trajectory_summary.get("baseline_metrics", {})
        if baseline:
            parts.append("\n### baseline metrics (trajectory + fractal proxies)\n"
                         "```json")
            parts.append(json.dumps(baseline, indent=2, default=str))
            parts.append("```")

        # -- Loss trend --
        parts.append("\n## LOSS HISTORY SUMMARY\n")
        parts.append("```json")
        parts.append(json.dumps(context.loss_history_summary, indent=2,
                                default=str))
        parts.append("```")

        # -- Current hparams --
        parts.append("\n## CURRENT configs/hparams.yaml\n```yaml")
        parts.append(context.hparams_yaml.rstrip("\n"))
        parts.append("```")

        # -- Previous attempts --
        if context.previous_attempts:
            parts.append("\n## PREVIOUS REPAIR ATTEMPTS (most recent last)\n")
            for a in context.previous_attempts:
                parts.append(f"- status={a.get('status', '?')} "
                             f"score_before={a.get('divergence_before', '?')} "
                             f"score_after={a.get('divergence_after', '?')} "
                             f"summary={a.get('summary', '?')}")

        # -- Constraints --
        parts.append("\n## CONSTRAINTS\n")
        parts.append(f"- only patchable file: {context.allowed_files[0]}")
        parts.append(f"- max lines changed: {context.max_lines_changed}")
        parts.append("- output ONLY the <<<PATCH ... >>>PATCH block "
                     "or `NO_FIX_FOUND: <reason>`")

        return "\n".join(parts)
