"""Context injection — pull penultimate activations from K nearest neighbors.

When a new expert spawns, we want to enrich its input with signals from
already-trained experts on the same data batch. This module produces
the auxiliary context tensor that ContextAwareMLP's context lane
consumes.

Aggregation is inverse-distance-weighted mean over the K neighbors'
penultimate activations — the weights come straight from the registry's
`composite_weights` softmax. Mean-pool keeps the context dimension
constant regardless of K (and regardless of later registry growth),
which is required for the probe-signature invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from fractaltrainer.integration.context_mlp import (
    ContextAwareMLP,
    PENULTIMATE_DIM,
)


@dataclass
class ContextSpec:
    """Parameters for gather_context.

    Attributes:
        k: number of nearest neighbors contributing context.
        aggregation: "weighted_mean" (inverse-distance-softmax weights)
            or "uniform_mean" (simple mean). Mean-pool either way.
        temperature: softmax temperature for weighted_mean; higher
            temperature → flatter weights. Ignored for uniform_mean.
    """

    k: int = 3
    aggregation: str = "weighted_mean"
    temperature: float = 1.0


def _as_tensor(probe: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(probe, np.ndarray):
        return torch.from_numpy(probe).float()
    return probe


def gather_context(
    neighbor_models: Sequence[ContextAwareMLP | torch.nn.Module],
    probe: torch.Tensor,
    spec: ContextSpec | None = None,
    distances: Sequence[float] | None = None,
    penultimate_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Aggregate the K neighbors' penultimate activations on the probe batch.

    Args:
        neighbor_models: list of K models whose penultimate activations
            on `probe` should be pooled. Each model must expose a
            `penultimate(x)` method returning a (B, 32) tensor, OR the
            caller must supply `penultimate_fn` to extract it.
        probe: (B, 1, 28, 28) or (B, 784) input batch.
        spec: ContextSpec controlling K and aggregation. Defaults to
            k=3, weighted_mean.
        distances: (K,) signature-space distances from the registry
            for each neighbor. Required for weighted_mean. If None and
            aggregation == weighted_mean, falls back to uniform mean.
        penultimate_fn: optional extractor for models that don't have a
            .penultimate() method (e.g. a legacy nn.Sequential MLP —
            you'd pass `lambda m, x: m.net[:-1](x.view(x.size(0), -1))`).

    Returns:
        (B, PENULTIMATE_DIM=32) float tensor on CPU, ready to be passed
        as `context` into ContextAwareMLP.forward().
    """
    spec = spec or ContextSpec()
    probe_t = _as_tensor(probe)
    out_device = probe_t.device

    if len(neighbor_models) == 0:
        batch = probe_t.size(0)
        return torch.zeros(batch, PENULTIMATE_DIM, device=out_device)

    k = min(spec.k, len(neighbor_models))
    models = list(neighbor_models[:k])

    penultimates: list[torch.Tensor] = []
    for m in models:
        m.eval()
        with torch.no_grad():
            if penultimate_fn is not None:
                p = penultimate_fn(m, probe_t)
            elif hasattr(m, "penultimate"):
                p = m.penultimate(probe_t)
            else:
                raise TypeError(
                    f"model {type(m).__name__} has no .penultimate() "
                    "method; pass penultimate_fn"
                )
        if p.ndim != 2 or p.size(1) != PENULTIMATE_DIM:
            raise ValueError(
                f"expected penultimate of shape (B, {PENULTIMATE_DIM}), "
                f"got {tuple(p.shape)}"
            )
        penultimates.append(p.float().to(out_device))

    stacked = torch.stack(penultimates, dim=0)  # (K, B, 32)

    if spec.aggregation == "uniform_mean" or distances is None:
        return stacked.mean(dim=0)

    if spec.aggregation == "weighted_mean":
        d = np.asarray(list(distances)[:k], dtype=np.float64)
        centered = -(d - d.min()) / max(spec.temperature, 1e-9)
        exp = np.exp(centered)
        w = exp / exp.sum()
        w_t = torch.from_numpy(w).float().view(k, 1, 1).to(out_device)
        return (stacked * w_t).sum(dim=0)

    raise ValueError(f"unknown aggregation: {spec.aggregation!r}")


def random_context(
    batch_size: int,
    seed: int = 0,
    scale: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a random (B, 32) context tensor for the ablation control arm.

    Arm C in the ablation: "random neighbors" — we simulate the
    activations of a randomly-selected (i.e. not-nearest) set of experts
    by drawing a normally-distributed tensor of matching shape. Used to
    test whether B > A is due to routing (signature-nearest) or just
    any auxiliary input.
    """
    g = torch.Generator().manual_seed(seed)
    out = torch.randn(batch_size, PENULTIMATE_DIM, generator=g) * scale
    if device is not None:
        out = out.to(device)
    return out
