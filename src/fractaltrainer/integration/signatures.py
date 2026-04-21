"""Signature functions — softmax (classification) and penultimate (latent).

Sprint 7b established that softmax-probe signatures on MNIST binary
tasks have ρ = -0.85 correlation with label-set Jaccard. That's the
current default used across Sprints 3-17 and the signature the
registry's `decide()` measures L2 distance in.

This module exposes both the existing softmax signature and a new
penultimate-layer latent signature. The latter is a candidate
alternative that:

- Works on non-classification regimes (regression — softmax is a
  category error there; RL — action distributions vary by task
  vocabulary).
- Carries a richer representation of "what the expert has learned"
  than a K-class probability vector.
- Aligns naturally with context injection (Sprint 17 C) — the
  penultimate is literally what gather_context pulls, so the
  signature space and the context space are the same space.

Signatures are expected to be swappable: the registry's routing is
just L2 nearest-neighbor search, which works for any fixed-dim
vector. The only discipline is that every entry in a registry must
have been signatured with the same function — mixing modes breaks
comparability.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


SignatureFn = Callable[[torch.nn.Module, torch.Tensor], np.ndarray]


def softmax_signature(model: torch.nn.Module, probe: torch.Tensor) -> np.ndarray:
    """Default Sprint-3+ signature: softmax over probe batch, flattened.

    Shape: `(batch × n_classes,)` — e.g. 100 × 10 = 1000 for the
    canonical MNIST probe + 10-class MLP.
    """
    model.eval()
    with torch.no_grad():
        try:
            logits = model(probe, context=None)
        except TypeError:
            logits = model(probe)
        p = F.softmax(logits, dim=1)
    return p.flatten().cpu().numpy()


def _raw_penultimate(model: torch.nn.Module, probe: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        if not hasattr(model, "penultimate"):
            raise TypeError(
                f"model {type(model).__name__} has no .penultimate() "
                "method; only ContextAwareMLP-style models are supported"
            )
        try:
            return model.penultimate(probe, context=None)
        except TypeError:
            return model.penultimate(probe)


def penultimate_signature(model: torch.nn.Module, probe: torch.Tensor) -> np.ndarray:
    """Raw penultimate-layer activations, flattened.

    Shape: `(batch × hidden_dim,)`. Note: validation on the Sprint 17
    registry showed this has no useful separation in L2 space
    (gap < 0, ρ ≈ 0.14) — raw penultimate magnitudes vary across
    seeds within the same task more than they vary across tasks. See
    `penultimate_normalized_signature` for a usable variant.
    """
    p = _raw_penultimate(model, probe)
    return p.flatten().cpu().numpy()


def penultimate_normalized_signature(
    model: torch.nn.Module, probe: torch.Tensor,
) -> np.ndarray:
    """L2-normalized penultimate signature (unit-norm direction).

    Same shape as `penultimate_signature` but with the full flattened
    vector projected to the unit sphere. This removes the scale variance
    that comes from random init + Adam trajectory differences across
    seeds, leaving only *direction* — which should be more task-
    determined if the penultimate carries any task identity at all.

    L2-normalization also bounds pairwise L2 distance to [0, 2], making
    distances directly comparable and match thresholds easier to
    calibrate.
    """
    p = _raw_penultimate(model, probe).flatten().cpu().numpy()
    n = float(np.linalg.norm(p))
    if n <= 1e-12:
        return p
    return p / n


def penultimate_softmax_signature(
    model: torch.nn.Module, probe: torch.Tensor,
) -> np.ndarray:
    """Row-wise softmax over penultimate activations, flattened.

    Treats each row of the penultimate batch (shape (hidden_dim,)) as
    a vector of pre-softmax logits, applies softmax, flattens. This
    gives a simplex-bounded signature with the same per-row structure
    as the softmax signature but in the hidden-layer space rather than
    the class-output space. Adapter concept for the generic-task
    registry.
    """
    p = _raw_penultimate(model, probe)
    p_sm = F.softmax(p, dim=1)
    return p_sm.flatten().cpu().numpy()


def get_signature_fn(mode: str) -> SignatureFn:
    """Look up a signature function by mode name.

    Known modes: "softmax" (default), "penultimate",
    "penultimate_normalized", "penultimate_softmax".
    """
    if mode == "softmax":
        return softmax_signature
    if mode == "penultimate":
        return penultimate_signature
    if mode == "penultimate_normalized":
        return penultimate_normalized_signature
    if mode == "penultimate_softmax":
        return penultimate_softmax_signature
    raise ValueError(
        f"unknown signature mode: {mode!r} (choose from "
        "'softmax', 'penultimate', 'penultimate_normalized', "
        "'penultimate_softmax')"
    )
