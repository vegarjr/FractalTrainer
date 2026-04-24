"""Hybrid head: meta-trained frozen encoder + trainable context-aware head.

Motivated by Sprint 19b's finding that plain FractalTrainer spawn loses
to ProtoNets by 18.7 pp on Omniglot 5-way 5-shot, while the context-
injection primitive independently validates at p=0.018. The diagnosis
was: the encoder (first two FC layers) is supervised from scratch on
25 support examples per episode, whereas ProtoNets meta-learns it
across thousands of background episodes.

This hybrid keeps FractalTrainer's registry + spawn + context mechanism
but replaces the encoder's from-scratch training with a frozen, pre-
meta-trained encoder. Only the final classification layer (and the
context lane projecting neighbor context into the first hidden) are
trainable per episode.

Implementation: load meta-trained weights into a ContextAwareMLP's
``fc1`` and ``fc2`` Linears and freeze them. ``fc3`` (head) and the
context lane (``ctx_proj`` + ``ctx_norm``) are re-initialised fresh per
episode and remain trainable. The softmax-signature contract is
preserved because forward() is structurally unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fractaltrainer.integration.context_mlp import (
    ContextAwareMLP,
    HIDDEN_DIM,
    INPUT_DIM,
    PENULTIMATE_DIM,
)


class MLPEncoderTrunk(nn.Module):
    """Two-layer trunk that matches ContextAwareMLP's fc1, fc2 structure.

    ``fc1``: INPUT_DIM (784) → HIDDEN_DIM (64)
    ``fc2``: HIDDEN_DIM (64) → PENULTIMATE_DIM (32)

    Forward returns the 32-d post-ReLU activation — the same quantity
    ContextAwareMLP.penultimate() returns. This is what ProtoNets
    meta-training should produce an embedding of.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, PENULTIMATE_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return h2


def load_pretrained_encoder(
    model: ContextAwareMLP,
    encoder: MLPEncoderTrunk | nn.Module,
    *,
    freeze: bool = True,
) -> ContextAwareMLP:
    """Copy meta-trained encoder weights into a ContextAwareMLP's fc1/fc2.

    Args:
        model: the fresh ContextAwareMLP that will become the hybrid expert.
        encoder: either an MLPEncoderTrunk (with fc1, fc2 attributes) or
            any nn.Module whose first two parameter blocks match fc1/fc2
            shapes. Anything else that lets us grab two (weight, bias)
            pairs in order works via state_dict shape-matching.
        freeze: if True, set ``requires_grad=False`` on fc1 and fc2. The
            head (fc3) and context lane (ctx_proj, ctx_norm) remain
            trainable.

    Returns:
        The same ``model`` for chaining.
    """
    if hasattr(encoder, "fc1") and hasattr(encoder, "fc2"):
        with torch.no_grad():
            model.fc1.weight.copy_(encoder.fc1.weight)
            model.fc1.bias.copy_(encoder.fc1.bias)
            model.fc2.weight.copy_(encoder.fc2.weight)
            model.fc2.bias.copy_(encoder.fc2.bias)
    else:
        raise TypeError(
            "encoder must expose .fc1 and .fc2 nn.Linear attributes; "
            f"got {type(encoder).__name__}"
        )

    if freeze:
        for p in model.fc1.parameters():
            p.requires_grad = False
        for p in model.fc2.parameters():
            p.requires_grad = False
    return model


def build_hybrid_expert(
    encoder: MLPEncoderTrunk,
    *,
    n_classes: int = 5,
    context_scale: float = 1.0,
    freeze_encoder: bool = True,
) -> ContextAwareMLP:
    """One-shot constructor: fresh ContextAwareMLP + loaded encoder + freeze.

    The returned model is ready to train per-episode: only fc3 and the
    context lane (ctx_proj + ctx_norm) have ``requires_grad=True``.
    """
    model = ContextAwareMLP(n_classes=n_classes, context_scale=context_scale)
    load_pretrained_encoder(model, encoder, freeze=freeze_encoder)
    return model


def trainable_parameters(model: ContextAwareMLP):
    """Return only the parameters with ``requires_grad=True``.

    Useful when constructing the optimizer for a hybrid expert — passing
    the full ``model.parameters()`` would include frozen fc1/fc2 weights
    which the optimizer would track needlessly.
    """
    return [p for p in model.parameters() if p.requires_grad]
