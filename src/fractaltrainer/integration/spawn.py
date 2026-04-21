"""Spawn wrapper — train a new expert with optional context-injection lane.

Two entry points:

    spawn_with_context(... neighbors, ContextSpec)   — Arm B of the ablation
    spawn_baseline(...)                              — Arm A (no context)

Both return a FractalEntry whose signature is computed by passing the
trained model over the canonical probe batch with `context=None` — see
the probe-signature invariant note in the Sprint 17 plan / Review.

The training loop is a thin torch.optim.Adam loop rather than going
through InstrumentedTrainer, because we need per-batch context
aggregation (neighbor penultimate activations on THAT batch's x, not
a fixed probe). InstrumentedTrainer's contract assumes a simple
`(x, y)` loader; wrapping it to inject context per-batch would require
monkey-patching its forward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fractaltrainer.integration.context_injection import (
    ContextSpec,
    gather_context,
    random_context,
)
from fractaltrainer.integration.context_mlp import (
    ContextAwareCNN,
    ContextAwareMLP,
    PENULTIMATE_DIM,
)


# Any nn.Module factory that returns a model with .penultimate() and
# accepts `context_scale` in its constructor works here. Default is the
# 784-d MLP used throughout Sprints 3-17.
ModelFactory = Callable[[float], torch.nn.Module]


def default_model_factory(context_scale: float) -> torch.nn.Module:
    return ContextAwareMLP(context_scale=context_scale)


def cnn_model_factory(context_scale: float) -> torch.nn.Module:
    return ContextAwareCNN(context_scale=context_scale)
from fractaltrainer.integration.signatures import (
    SignatureFn,
    get_signature_fn,
    softmax_signature,
)
from fractaltrainer.registry import FractalEntry


@dataclass
class TrainStats:
    n_steps: int
    final_loss: float
    loss_history: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0


def _probe_signature(
    model: torch.nn.Module,
    probe: torch.Tensor,
    signature_fn: SignatureFn | None = None,
) -> np.ndarray:
    """Compute a signature for this model on the probe batch.

    Defaults to the Sprint-3 softmax signature. Pass a different
    `signature_fn` (e.g. `penultimate_signature`) to change the
    signature space — caller is responsible for ensuring every
    entry in a registry uses the same signature_fn for comparability.
    """
    fn = signature_fn or softmax_signature
    return fn(model, probe)


def _train_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    context: torch.Tensor | None,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(x, context=context)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _context_for_batch(
    x: torch.Tensor,
    mode: str,
    neighbors: Sequence[nn.Module] | None,
    neighbor_distances: Sequence[float] | None,
    spec: ContextSpec | None,
    rng_seed: int,
    step: int,
) -> torch.Tensor | None:
    """Return the (B, 32) context to use for this training batch.

    mode:
        "none"      — return None (baseline arm)
        "neighbors" — gather_context from nearest-K neighbors (arm B)
        "random"    — draw a fresh random tensor per step (arm C)
    """
    if mode == "none":
        return None
    if mode == "random":
        return random_context(
            batch_size=x.size(0), seed=rng_seed + step, scale=1.0,
        )
    if mode == "neighbors":
        if not neighbors:
            return None
        return gather_context(
            list(neighbors), x, spec or ContextSpec(),
            distances=neighbor_distances,
        )
    raise ValueError(f"unknown context mode: {mode!r}")


def _train_loop(
    model: torch.nn.Module,
    dataloader: Iterable,
    *,
    n_steps: int,
    lr: float,
    context_mode: str,
    neighbors: Sequence[nn.Module] | None,
    neighbor_distances: Sequence[float] | None,
    spec: ContextSpec | None,
    rng_seed: int = 0,
) -> TrainStats:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[float] = []
    step = 0
    t0 = time.time()
    iterator = iter(dataloader)
    while step < n_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        x, y = batch
        context = _context_for_batch(
            x, context_mode, neighbors, neighbor_distances, spec,
            rng_seed, step,
        )
        loss = _train_step(model, x, y, context, optimizer)
        loss_history.append(loss)
        step += 1
    elapsed = time.time() - t0
    return TrainStats(
        n_steps=step,
        final_loss=loss_history[-1] if loss_history else float("nan"),
        loss_history=loss_history,
        elapsed_s=elapsed,
    )


def spawn_baseline(
    dataloader: Iterable,
    probe: torch.Tensor,
    *,
    n_steps: int = 500,
    lr: float = 0.01,
    seed: int = 42,
    entry_name: str = "spawn_baseline",
    task: str | None = None,
    metadata_extra: dict | None = None,
    signature_fn: SignatureFn | None = None,
    model_factory: ModelFactory | None = None,
) -> tuple[torch.nn.Module, FractalEntry, TrainStats]:
    """Train a fresh model (default ContextAwareMLP) with the context lane disabled.

    Arm A of the ablation — the no-context control. Returns the model,
    its FractalEntry (signature + metadata), and training stats.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    factory = model_factory or default_model_factory
    model = factory(0.0)
    stats = _train_loop(
        model, dataloader,
        n_steps=n_steps, lr=lr,
        context_mode="none",
        neighbors=None, neighbor_distances=None, spec=None,
        rng_seed=seed,
    )
    sig = _probe_signature(model, probe, signature_fn=signature_fn)
    meta = {
        "task": task, "seed": seed, "spawned": True,
        "context_mode": "none", "train_wall_s": stats.elapsed_s,
    }
    if metadata_extra:
        meta.update(metadata_extra)
    entry = FractalEntry(name=entry_name, signature=sig, metadata=meta)
    return model, entry, stats


def spawn_with_context(
    dataloader: Iterable,
    probe: torch.Tensor,
    *,
    neighbors: Sequence[nn.Module],
    neighbor_distances: Sequence[float] | None = None,
    spec: ContextSpec | None = None,
    context_scale: float = 1.0,
    n_steps: int = 500,
    lr: float = 0.01,
    seed: int = 42,
    entry_name: str = "spawn_with_context",
    task: str | None = None,
    metadata_extra: dict | None = None,
    signature_fn: SignatureFn | None = None,
    model_factory: ModelFactory | None = None,
) -> tuple[torch.nn.Module, FractalEntry, TrainStats]:
    """Train a fresh ContextAwareMLP with the context lane enabled.

    Arm B of the ablation. Each training step pulls the aggregated
    penultimate activations from `neighbors` on that step's input
    batch, normalizes + projects them, and adds them to the first
    hidden layer. Probe signature is computed with context=None so the
    resulting entry can be compared against legacy entries under the
    same distance metric.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    factory = model_factory or default_model_factory
    model = factory(context_scale)
    stats = _train_loop(
        model, dataloader,
        n_steps=n_steps, lr=lr,
        context_mode="neighbors",
        neighbors=neighbors,
        neighbor_distances=neighbor_distances,
        spec=spec or ContextSpec(),
        rng_seed=seed,
    )
    sig = _probe_signature(model, probe, signature_fn=signature_fn)
    meta = {
        "task": task, "seed": seed, "spawned": True,
        "context_mode": "neighbors",
        "context_k": (spec or ContextSpec()).k,
        "context_scale": float(context_scale),
        "n_neighbors": len(neighbors),
        "train_wall_s": stats.elapsed_s,
    }
    if metadata_extra:
        meta.update(metadata_extra)
    entry = FractalEntry(name=entry_name, signature=sig, metadata=meta)
    return model, entry, stats


def spawn_random_context(
    dataloader: Iterable,
    probe: torch.Tensor,
    *,
    context_scale: float = 1.0,
    n_steps: int = 500,
    lr: float = 0.01,
    seed: int = 42,
    entry_name: str = "spawn_random_context",
    task: str | None = None,
    metadata_extra: dict | None = None,
    signature_fn: SignatureFn | None = None,
    model_factory: ModelFactory | None = None,
) -> tuple[torch.nn.Module, FractalEntry, TrainStats]:
    """Train with a FRESH random context tensor each step.

    Arm C of the ablation — context is random noise, not neighbor
    activations. Isolates "context from routing" vs "context from
    anywhere". If this arm matches Arm B, routing is doing nothing.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    factory = model_factory or default_model_factory
    model = factory(context_scale)
    stats = _train_loop(
        model, dataloader,
        n_steps=n_steps, lr=lr,
        context_mode="random",
        neighbors=None, neighbor_distances=None, spec=None,
        rng_seed=seed,
    )
    sig = _probe_signature(model, probe, signature_fn=signature_fn)
    meta = {
        "task": task, "seed": seed, "spawned": True,
        "context_mode": "random",
        "context_scale": float(context_scale),
        "train_wall_s": stats.elapsed_s,
    }
    if metadata_extra:
        meta.update(metadata_extra)
    entry = FractalEntry(name=entry_name, signature=sig, metadata=meta)
    return model, entry, stats
