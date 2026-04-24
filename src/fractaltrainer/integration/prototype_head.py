"""Prototype-based expert: frozen encoder + per-class prototypes.

Sprint 19c's hybrid (meta-trained frozen encoder + trainable softmax
head) recovers 12.6 pp of the 18.7 pp Omniglot gap but still loses to
ProtoNets by 5.6 pp. The residual gap is consistent with the mismatch
between meta-training (ProtoNets-style, joint with prototype classifier)
and test-time inference (freeze-then-finetune with softmax head).

This module closes that loop: use the same meta-trained encoder, but at
classification time use **prototype-distance** instead of a trained
softmax. The "expert" now has zero trainable parameters at spawn time —
it's just a snapshot of support-set prototypes.

``PrototypeExpert`` intentionally wears the same duck-typed interface as
``ContextAwareMLP`` (``.forward(x, context=None)``, ``.penultimate(x)``,
``.eval()``) so existing ``softmax_signature``, ``gather_context``, and
``FractalRegistry`` routing machinery work unchanged.

Spawn contract: ``PrototypeExpert.fit(encoder, support_x, support_y)``
computes prototypes from the support set and returns a frozen expert.
The registry's signature is computed by passing a fixed probe batch
through the expert's forward (logits = −distance / temperature) and
softmax-ing. At query time, argmax of −distance to prototypes gives
the prediction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PrototypeExpert(nn.Module):
    """A classifier defined entirely by support-set prototypes.

    Attributes:
        encoder: a frozen, meta-trained feature extractor. Typically an
            ``MLPEncoderTrunk`` trained by ProtoNets on a background set.
        prototypes: a ``(n_way, d_emb)`` buffer — one prototype per class.
        temperature: softmax temperature on the negative-distance logits.
            Larger = softer; used to produce non-saturated signatures.
    """

    def __init__(self, encoder: nn.Module, prototypes: torch.Tensor,
                 temperature: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("prototypes", prototypes)
        self.temperature = float(temperature)

    # --- ContextAwareMLP-duck-typed interface -----------------------
    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        """Returns logits = −distance / temperature. Shape (B, n_way).

        The ``context`` argument is accepted for interface compatibility
        with ``gather_context`` callers but ignored — a prototype
        expert has no trainable parameters for context to influence.
        """
        emb = self._embed(x)
        d = torch.cdist(emb, self.prototypes)
        return -d / self.temperature

    def penultimate(self, x: torch.Tensor, context=None) -> torch.Tensor:
        """The encoder's 32-d output is the natural 'penultimate' — it's
        what context injection into a ContextAwareMLP would later consume.
        """
        return self._embed(x)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)

    # --- Factory from support ---------------------------------------
    @classmethod
    def fit(
        cls,
        encoder: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *,
        n_way: int,
        temperature: float = 1.0,
    ) -> "PrototypeExpert":
        """Compute per-class prototypes from support and return a frozen
        expert. `support_y` must contain every class in 0..n_way-1 at
        least once, otherwise the corresponding prototype would be NaN.
        """
        encoder.eval()
        if support_x.ndim > 2:
            support_x = support_x.view(support_x.size(0), -1)
        with torch.no_grad():
            emb = encoder(support_x)
        protos = []
        for k in range(n_way):
            mask = support_y == k
            if not mask.any():
                raise ValueError(
                    f"class {k} has no support examples; cannot compute prototype"
                )
            protos.append(emb[mask].mean(0))
        protos_t = torch.stack(protos)  # (n_way, d_emb)
        return cls(encoder, protos_t, temperature=temperature)

    # --- Convenience ------------------------------------------------
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=1)

    def score(self, x: torch.Tensor, y: torch.Tensor) -> float:
        pred = self.predict(x)
        return float((pred == y).float().mean().item())


def compose_prototypes(
    experts: list[PrototypeExpert],
    weights: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend multiple experts' prototype tensors.

    Useful when the registry returns K nearest same-task entries at
    match/compose verdict: the blended prototype tensor is a more
    stable estimate than any single expert's prototypes.

    Args:
        experts: list of K PrototypeExperts with matching prototype shapes.
        weights: (K,) float weights, defaults to uniform. Will be
            normalized to sum to 1.

    Returns:
        (n_way, d_emb) tensor — the weighted-mean prototype block.
    """
    if not experts:
        raise ValueError("need at least one expert")
    protos = torch.stack([e.prototypes for e in experts], dim=0)  # (K, n_way, d_emb)
    if weights is None:
        w = torch.full((len(experts),), 1.0 / len(experts),
                       dtype=protos.dtype, device=protos.device)
    else:
        w_np = np.asarray(weights, dtype=np.float64)
        w_np = w_np / w_np.sum()
        w = torch.from_numpy(w_np).to(dtype=protos.dtype, device=protos.device)
    return (protos * w.view(-1, 1, 1)).sum(dim=0)
