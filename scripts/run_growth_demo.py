"""v3 Sprint 4 — Growth-trigger demo.

Demonstrates the three verdicts of FractalRegistry.decide end-to-end:

  A. MATCH   — query is a known task in the registry → route to nearest.
  B. SPAWN   — query is a genuinely new task, no close neighbor → train
               a new expert, register it, re-query (now matches).
  C. COMPOSE — query sits in the ambiguity band between match and spawn
               → take top-K weighted blend.

Threshold calibration from v3 Sprint 3 MVP (Reviews/12):
    Same-task (MNIST): distances 0.93 to 3.43
    Same-task (Fashion, OOD probe): 6.68
    Cross-task: distances 7.32 to 10.80+
    Clean gap: (3.43, 7.32) for MNIST, narrower for Fashion.

Using:
    match_threshold  = 5.0   — catches MNIST same-task; Fashion falls in compose
    spawn_threshold  = 7.0   — below cross-task minimum; genuinely new → spawn

For scenario B we actually train a new expert for a task not yet in the
registry (y ∈ {1, 4, 9} = perfect_squares). The spawn path is exercised
end-to-end: decide → train → register → re-query matches.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.observer.trainer import InstrumentedTrainer  # noqa: E402
from fractaltrainer.registry import (  # noqa: E402
    FractalEntry,
    FractalRegistry,
)


LAYER_SHAPES = [
    ((64, 784), "net.0.weight"), ((64,), "net.0.bias"),
    ((32, 64),  "net.2.weight"), ((32,), "net.2.bias"),
    ((10, 32),  "net.4.weight"), ((10,), "net.4.bias"),
]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def _load_model_from_traj(traj_path: Path) -> nn.Module:
    traj = np.load(traj_path)
    final_flat = traj[-1]
    model = MLP()
    offset = 0
    sd = {}
    for shape, name in LAYER_SHAPES:
        size = int(np.prod(shape))
        chunk = final_flat[offset:offset + size]
        sd[name] = torch.tensor(chunk.reshape(shape), dtype=torch.float32)
        offset += size
    model.load_state_dict(sd)
    model.eval()
    return model


def _probe(model: nn.Module, probe: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(probe)
        probs = F.softmax(logits, dim=1)
    return probs.flatten().cpu().numpy()


def _mnist_probe(data_dir: str, n: int, seed: int) -> torch.Tensor:
    from torchvision import datasets, transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(base), size=n, replace=False)
    return torch.stack([base[i][0] for i in idx.tolist()], dim=0)


def _relabel(y: int, task: str) -> int:
    if task == "digit_class":        return int(y)
    if task == "parity":             return int(y % 2)
    if task == "high_vs_low":        return int(y >= 5)
    if task == "mod3":               return int(y % 3)
    if task == "mod5":               return int(y % 5)
    if task == "primes_vs_rest":     return int(y in (2, 3, 5, 7))
    if task == "ones_vs_teens":      return int(y <= 4)
    if task == "triangular":         return int(y in (1, 3, 6))
    if task == "fibonacci":          return int(y in (1, 2, 3, 5, 8))
    if task == "middle_456":         return int(y in (4, 5, 6))
    if task == "fashion_class":      return int(y)
    if task == "perfect_squares":    return int(y in (1, 4, 9))
    raise ValueError(task)


class RelabeledMNIST(Dataset):
    def __init__(self, base, task): self.base, self.task = base, task
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, _relabel(int(y), self.task)


def _make_train_loader(task: str, train_size: int, batch_size: int,
                        data_dir: str, seed: int) -> DataLoader:
    from torchvision import datasets, transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    ds = RelabeledMNIST(base, task)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=train_size, replace=False)
    return DataLoader(Subset(ds, idx.tolist()),
                       batch_size=batch_size, shuffle=True, drop_last=True)


def spawn_new_fractal(
    registry: FractalRegistry,
    task: str,
    seed: int,
    probe: torch.Tensor,
    data_dir: str,
    out_dir: str,
    n_steps: int = 500,
    train_size: int = 5000,
    batch_size: int = 64,
    snapshot_every: int = 10,
) -> tuple[FractalEntry, float]:
    """Train a new expert from scratch, build its activation signature,
    add it to the registry, return the new entry + training wall-clock.
    """
    hparams = {
        "learning_rate": 0.01, "batch_size": batch_size,
        "weight_decay": 0.0, "dropout": 0.0,
        "init_seed": seed, "optimizer": "adam",
    }
    model = MLP()
    loader = _make_train_loader(task, train_size, batch_size, data_dir, seed)
    run_id = f"spawn_{task}_seed{seed}"
    trainer = InstrumentedTrainer(
        model=model, dataloader=loader, loss_fn=F.cross_entropy,
        hparams=hparams, snapshot_every=snapshot_every,
        out_dir=out_dir, run_id=run_id,
    )
    t0 = time.time()
    run = trainer.train(n_steps)
    elapsed = time.time() - t0

    sig = _probe(model, probe)
    entry = FractalEntry(
        name=f"{task}_seed{seed}", signature=sig,
        metadata={
            "task": task, "seed": seed,
            "trajectory_path": run.snapshot_path,
            "spawned": True, "train_wall_s": elapsed,
        },
    )
    registry.add(entry)
    return entry, elapsed


def _load_cached_entries(
    traj_dir: Path, tasks: list[str], seeds: list[int],
    probe: torch.Tensor,
) -> dict[tuple[str, int], FractalEntry]:
    out = {}
    for task in tasks:
        for seed in seeds:
            p = traj_dir / f"ext_{task}_seed{seed}_trajectory.npy"
            if not p.is_file():
                continue
            model = _load_model_from_traj(p)
            sig = _probe(model, probe)
            out[(task, seed)] = FractalEntry(
                name=f"{task}_seed{seed}",
                signature=sig,
                metadata={"task": task, "seed": seed,
                          "trajectory_path": str(p),
                          "spawned": False},
            )
    return out


MNIST_TASKS = [
    "digit_class", "parity", "high_vs_low",
    "mod3", "mod5", "primes_vs_rest", "ones_vs_teens",
    "triangular", "fibonacci", "middle_456",
]
FASHION_TASKS = ["fashion_class"]
CACHED_TASKS = MNIST_TASKS + FASHION_TASKS
CACHED_SEEDS = [42, 101, 2024]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--spawn-out-dir", type=str,
                        default="results/growth_spawn_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--match-threshold", type=float, default=5.0)
    parser.add_argument("--spawn-threshold", type=float, default=7.0)
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--results-out", type=str,
                        default="results/growth_demo.json")
    args = parser.parse_args(argv)

    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)
    all_cached = _load_cached_entries(
        Path(args.traj_dir), CACHED_TASKS, CACHED_SEEDS, probe,
    )
    print(f"Cached entries loaded: {len(all_cached)}")
    print(f"Thresholds: match ≤ {args.match_threshold}, "
          f"compose ≤ {args.spawn_threshold}, else spawn.")

    scenarios: list[dict] = []

    # ── Scenario A: MATCH — known task in full registry ─────────
    print("\n" + "=" * 72)
    print("  SCENARIO A — MATCH")
    print("=" * 72)
    reg_a = FractalRegistry()
    for (t, s), e in all_cached.items():
        if s in (42, 101):
            reg_a.add(e)
    # Query: the seed=2024 entry of a known task
    query_entry = all_cached[("parity", 2024)]
    d = reg_a.decide(
        query_entry.signature,
        match_threshold=args.match_threshold,
        spawn_threshold=args.spawn_threshold,
        compose_k=3,
    )
    print(f"  registry size: {len(reg_a)}")
    print(f"  query: {query_entry.name}")
    print(f"  verdict: {d.verdict}   min_distance={d.min_distance:.3f}")
    print(f"  nearest: {d.retrieval.nearest.name} "
          f"(d={d.retrieval.distances[0]:.3f})")
    scenarios.append({
        "scenario": "A_match",
        "registry_size": len(reg_a),
        "query": query_entry.name,
        "decision": d.to_dict(),
    })
    assert d.verdict == "match", f"expected match, got {d.verdict}"

    # ── Scenario B: SPAWN — novel task, no close neighbor ───────
    print("\n" + "=" * 72)
    print("  SCENARIO B — SPAWN (new task: perfect_squares = y ∈ {1, 4, 9})")
    print("=" * 72)
    # Fresh registry with only the seed=42 + 101 entries (no perfect_squares)
    reg_b = FractalRegistry()
    for (t, s), e in all_cached.items():
        if s in (42, 101):
            reg_b.add(e)
    # Step 1: train a probe-fractal for perfect_squares (what a user would do
    # when they have data for a new task but no registered expert)
    print(f"  registry size: {len(reg_b)}")
    print("  training a perfect_squares expert to compute its signature...")
    # Instead of spawning directly, first train once to get the signature,
    # then use that signature to call decide.
    probe_entry, train_s = spawn_new_fractal(
        registry=FractalRegistry(),  # don't add yet — just train
        task="perfect_squares", seed=42, probe=probe,
        data_dir=args.data_dir, out_dir=args.spawn_out_dir,
    )
    print(f"    trained in {train_s:.1f}s")
    d = reg_b.decide(
        probe_entry.signature,
        match_threshold=args.match_threshold,
        spawn_threshold=args.spawn_threshold,
        compose_k=3,
    )
    print(f"  decision BEFORE spawn: verdict={d.verdict}   "
          f"min_distance={d.min_distance:.3f}")
    print(f"  nearest (rejected): {d.retrieval.nearest.name} "
          f"(d={d.retrieval.distances[0]:.3f})")
    assert d.verdict == "spawn", f"expected spawn, got {d.verdict}"

    # Step 2: execute the spawn (register the already-trained expert)
    reg_b.add(probe_entry)
    print(f"  → REGISTERED: {probe_entry.name}  "
          f"(registry size now {len(reg_b)})")

    # Step 3: re-query — should now match
    d2 = reg_b.decide(
        probe_entry.signature,
        match_threshold=args.match_threshold,
        spawn_threshold=args.spawn_threshold,
        compose_k=3,
    )
    print(f"  decision AFTER spawn:  verdict={d2.verdict}   "
          f"min_distance={d2.min_distance:.3f}   "
          f"matched: {d2.retrieval.nearest.name}")
    assert d2.verdict == "match", f"expected match after spawn, got {d2.verdict}"
    scenarios.append({
        "scenario": "B_spawn",
        "registry_size_before": 22,
        "registry_size_after": len(reg_b),
        "new_task": "perfect_squares",
        "train_wall_s": train_s,
        "decision_before_spawn": d.to_dict(),
        "decision_after_spawn": d2.to_dict(),
    })

    # ── Scenario C: COMPOSE — Fashion-MNIST against MNIST-only registry ──
    print("\n" + "=" * 72)
    print("  SCENARIO C — COMPOSE (Fashion vs MNIST-only registry)")
    print("=" * 72)
    # Register only MNIST entries (no fashion_class). Query with fashion.
    reg_c = FractalRegistry()
    for (t, s), e in all_cached.items():
        if t == "fashion_class":
            continue
        if s in (42, 101, 2024):
            reg_c.add(e)
    query_entry = all_cached[("fashion_class", 42)]
    d = reg_c.decide(
        query_entry.signature,
        match_threshold=args.match_threshold,
        spawn_threshold=args.spawn_threshold,
        compose_k=3,
    )
    print(f"  registry size: {len(reg_c)} (MNIST-only, no Fashion)")
    print(f"  query: {query_entry.name}")
    print(f"  verdict: {d.verdict}   min_distance={d.min_distance:.3f}")
    if d.verdict == "compose":
        print(f"  top-K composite:")
        for e, w in zip(d.composite_entries, d.composite_weights):
            print(f"    {e.name:<30s}  weight={w:.3f}")
    elif d.verdict == "spawn":
        print(f"  nearest (would be rejected for spawn): "
              f"{d.retrieval.nearest.name} (d={d.retrieval.distances[0]:.3f})")
    scenarios.append({
        "scenario": "C_compose_or_spawn",
        "registry_size": len(reg_c),
        "query": query_entry.name,
        "decision": d.to_dict(),
    })

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for s in scenarios:
        name = s["scenario"]
        v = s.get("decision", s.get("decision_before_spawn", {})).get("verdict")
        extra = (f" → {s['decision_after_spawn']['verdict']} after spawn"
                 if "decision_after_spawn" in s else "")
        print(f"  {name:<24s}  verdict={v}{extra}")

    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "match_threshold": args.match_threshold,
            "spawn_threshold": args.spawn_threshold,
            "scenarios": scenarios,
        }, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
