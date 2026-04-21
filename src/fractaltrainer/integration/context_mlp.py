"""ContextAwareMLP — baseline MLP with optional penultimate-activation context lane.

Shape matches the Sprint 4 MLP exactly (784→64→32→10) so signatures
computed on a fixed probe batch are comparable across context-trained
and legacy experts. Context enters via an additive lane:

    h0   = Linear(784, 64)(x)
    c    = LayerNorm(32)(context)           # context aggregated by caller
    h0  += context_scale * Linear(32, 64)(c)
    h1   = ReLU(h0)
    h2   = ReLU(Linear(64, 32)(h1))
    logits = Linear(32, 10)(h2)

With `context_scale=0` the forward is numerically identical to a plain
MLP (the context lane contributes zero), which is the clean control arm
for the ablation.

The `penultimate(x)` method returns the (B, 32) tensor right before
the final Linear — that's what `gather_context` pulls from the nearest
K neighbors at spawn time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


PENULTIMATE_DIM = 32
HIDDEN_DIM = 64
INPUT_DIM = 784
N_CLASSES = 10


class ContextAwareMLP(nn.Module):
    """MLP with an optional context lane fused into the first hidden layer.

    Args:
        context_dim: dimension of the context tensor passed to `forward`.
            Defaults to PENULTIMATE_DIM (=32). Set to 0 to disable the
            context lane entirely (the Linear(context_dim, 64) module
            is not constructed).
        context_scale: multiplicative weight on the context lane's
            contribution. 0.0 makes the forward bit-exact to the
            baseline MLP.

    Attributes:
        fc1, fc2, fc3: the three Linear layers of the 784→64→32→10 spine.
        ctx_proj: Linear(context_dim, 64) projecting aggregated context
            into the first hidden space. None if `context_dim == 0`.
        ctx_norm: LayerNorm(context_dim) applied to context before ctx_proj.
    """

    def __init__(
        self,
        context_dim: int = PENULTIMATE_DIM,
        context_scale: float = 1.0,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.context_scale = float(context_scale)

        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, PENULTIMATE_DIM)
        self.fc3 = nn.Linear(PENULTIMATE_DIM, N_CLASSES)

        if self.context_dim > 0:
            self.ctx_norm = nn.LayerNorm(self.context_dim)
            self.ctx_proj = nn.Linear(self.context_dim, HIDDEN_DIM)
        else:
            self.ctx_norm = None
            self.ctx_proj = None

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return x

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._flatten(x)
        h0 = self.fc1(x)
        if (
            context is not None
            and self.ctx_proj is not None
            and self.context_scale != 0.0
        ):
            c = self.ctx_norm(context)
            h0 = h0 + self.context_scale * self.ctx_proj(c)
        h1 = torch.relu(h0)
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2)

    def penultimate(
        self, x: torch.Tensor, context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the (B, 32) activation right before the final Linear.

        Used by `gather_context` to pull neighbor activations into a
        new expert's training input. Mirrors the forward path exactly
        up to and including the second ReLU.
        """
        x = self._flatten(x)
        h0 = self.fc1(x)
        if (
            context is not None
            and self.ctx_proj is not None
            and self.context_scale != 0.0
        ):
            c = self.ctx_norm(context)
            h0 = h0 + self.context_scale * self.ctx_proj(c)
        h1 = torch.relu(h0)
        return torch.relu(self.fc2(h1))


def baseline_mlp_forward(model: ContextAwareMLP, x: torch.Tensor) -> torch.Tensor:
    """Reference forward of the baseline (no-context) MLP.

    Used by `test_context_mlp` to confirm that `context_scale=0` or
    `context=None` produces identical output. Exposed as a module-level
    helper so the test can call it without subclassing.
    """
    x = model._flatten(x)
    h1 = torch.relu(model.fc1(x))
    h2 = torch.relu(model.fc2(h1))
    return model.fc3(h2)
