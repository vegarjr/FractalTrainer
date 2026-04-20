"""v3 Sprint 7 — Scale stress test.

Goal: push the registry from 33 entries / 11 tasks to 93 entries /
31 tasks, and check whether the Sprint 3-6 invariants hold at 3× scale.

  1. Retrieval still at or near 100% same-task top-1?
  2. Calibrated thresholds still produce a clean within/cross gap?
  3. Cross-task distance distribution richness — any degenerate
     near-duplicate cases across tasks?
  4. decide() latency (naive O(N) scan) acceptable at N=93?

New tasks: 20 deterministically-sampled novel binary labelings of
MNIST digits. Each task is `y ∈ S` where S is a 3- to 6-digit
subset. Deterministic via a fixed PRNG seed; ensured non-colliding
with the 7 existing binary tasks (parity, high_vs_low, primes_vs_rest,
ones_vs_teens, triangular, fibonacci, middle_456) and their
complements.

Training: 500-step MNIST MLP (784→64→32→10), same hparams as Sprint
4 spawn (lr=0.01, bs=64, adam, weight_decay=0, dropout=0).
Approximate cost: ~8s per expert × 60 experts ≈ 8 minutes.

Artifacts:
  results/sprint7_v3_trajectories/ext_subset_<digits>_seed<K>_trajectory.npy
  results/sprint7_v3_scale_stress.json  (calibration + retrieval report)
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
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
from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402


LAYER_SHAPES = [
    ((64, 784), "net.0.weight"), ((64,), "net.0.bias"),
    ((32, 64),  "net.2.weight"), ((32,), "net.2.bias"),
    ((10, 32),  "net.4.weight"), ((10,), "net.4.bias"),
]


EXISTING_TASKS = [
    "digit_class", "parity", "high_vs_low", "mod3", "mod5",
    "primes_vs_rest", "ones_vs_teens", "triangular",
    "fibonacci", "middle_456", "fashion_class",
]
EXISTING_BINARY_SUBSETS = [
    frozenset({1, 3, 5, 7, 9}),       # parity
    frozenset({5, 6, 7, 8, 9}),       # high_vs_low
    frozenset({2, 3, 5, 7}),          # primes_vs_rest
    frozenset({0, 1, 2, 3, 4}),       # ones_vs_teens
    frozenset({1, 3, 6}),             # triangular
    frozenset({1, 2, 3, 5, 8}),       # fibonacci
    frozenset({4, 5, 6}),             # middle_456
]
SEEDS = [42, 101, 2024]


def _is_novel(subset: frozenset[int]) -> bool:
    """True if subset is not one of the existing binary tasks (or its
    complement — same decision boundary up to a label flip)."""
    full = frozenset(range(10))
    for ex in EXISTING_BINARY_SUBSETS:
        if subset == ex or subset == (full - ex):
            return False
    return True


def generate_novel_binary_tasks(n_tasks: int, seed: int = 42
                                 ) -> dict[str, tuple[int, ...]]:
    """Sample n_tasks novel binary labelings of MNIST digits.

    Subset sizes range 3–6 (avoids trivial 1/9 and 50/50 parity-like
    structures). Deterministic via `seed`.
    """
    candidates: list[tuple[int, ...]] = []
    for k in (3, 4, 5, 6):
        for c in itertools.combinations(range(10), k):
            if _is_novel(frozenset(c)):
                candidates.append(tuple(sorted(c)))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = candidates[:n_tasks]
    out: dict[str, tuple[int, ...]] = {}
    for subset in chosen:
        name = "subset_" + "".join(str(d) for d in subset)
        out[name] = subset
    return out


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
    trajectory = np.load(traj_path)
    final_flat = trajectory[-1]
    model = MLP()
    offset = 0
    state_dict = {}
    for shape, name in LAYER_SHAPES:
        size = int(np.prod(shape))
        chunk = final_flat[offset:offset + size]
        state_dict[name] = torch.tensor(chunk.reshape(shape), dtype=torch.float32)
        offset += size
    model.load_state_dict(state_dict)
    model.eval()
    return model


class SubsetRelabeledMNIST(Dataset):
    """y → 1 if y ∈ target_subset else 0."""
    def __init__(self, base, target_subset: set[int]):
        self.base, self.target = base, set(target_subset)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(int(y) in self.target)


def _train_subset_expert(
    subset: tuple[int, ...], seed: int,
    data_dir: str, out_dir: str,
    n_steps: int = 500, train_size: int = 5000,
    batch_size: int = 64, snapshot_every: int = 10,
) -> tuple[Path, float]:
    from torchvision import datasets, transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    ds = SubsetRelabeledMNIST(base, set(subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=train_size, replace=False)
    loader = DataLoader(Subset(ds, idx.tolist()),
                         batch_size=batch_size, shuffle=True, drop_last=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    hparams = {
        "learning_rate": 0.01, "batch_size": batch_size,
        "weight_decay": 0.0, "dropout": 0.0,
        "init_seed": seed, "optimizer": "adam",
    }
    run_id = f"ext_subset_{''.join(str(d) for d in subset)}_seed{seed}"
    trainer = InstrumentedTrainer(
        model=model, dataloader=loader, loss_fn=F.cross_entropy,
        hparams=hparams, snapshot_every=snapshot_every,
        out_dir=out_dir, run_id=run_id,
    )
    t0 = time.time()
    run = trainer.train(n_steps)
    elapsed = time.time() - t0
    return Path(run.snapshot_path), elapsed


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


def _signature(model: nn.Module, probe: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        probs = F.softmax(model(probe), dim=1)
    return probs.flatten().cpu().numpy()


def _load_cached_entries(
    traj_dir: Path, tasks: list[str], seeds: list[int], probe: torch.Tensor,
) -> list[FractalEntry]:
    out = []
    for task in tasks:
        for seed in seeds:
            p = traj_dir / f"ext_{task}_seed{seed}_trajectory.npy"
            if not p.is_file():
                continue
            model = _load_model_from_traj(p)
            sig = _signature(model, probe)
            out.append(FractalEntry(
                name=f"{task}_seed{seed}", signature=sig,
                metadata={"task": task, "seed": seed,
                          "trajectory_path": str(p),
                          "source": "cached"},
            ))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-traj-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--new-traj-dir", type=str,
                        default="results/sprint7_v3_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-new-tasks", type=int, default=20)
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--task-sample-seed", type=int, default=42,
                        help="PRNG seed for new-task subset selection")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--skip-training", action="store_true",
                        help="Assume new-traj-dir is already populated")
    parser.add_argument("--results-out", type=str,
                        default="results/sprint7_v3_scale_stress.json")
    args = parser.parse_args(argv)

    # Generate the novel tasks
    new_tasks = generate_novel_binary_tasks(
        n_tasks=args.n_new_tasks, seed=args.task_sample_seed,
    )
    print(f"Novel binary tasks ({len(new_tasks)}):")
    for name, subset in new_tasks.items():
        print(f"  {name:<20s}  y ∈ {set(subset)}")

    new_traj_dir = Path(args.new_traj_dir)
    new_traj_dir.mkdir(parents=True, exist_ok=True)

    # Train new experts (or reuse if present)
    train_log: list[dict] = []
    if not args.skip_training:
        print(f"\nTraining {len(new_tasks)} × {len(SEEDS)} = "
              f"{len(new_tasks) * len(SEEDS)} new experts...")
        t0 = time.time()
        for i, (name, subset) in enumerate(new_tasks.items(), start=1):
            for seed in SEEDS:
                traj_path = new_traj_dir / f"{name.replace('subset_', 'ext_subset_')}_seed{seed}_trajectory.npy"
                if traj_path.is_file():
                    print(f"  [{i:>2}/{len(new_tasks)}] {name} seed={seed}  "
                          f"(cached, skipping)")
                    train_log.append({"task": name, "seed": seed,
                                       "traj_path": str(traj_path),
                                       "elapsed_s": 0.0, "skipped": True})
                    continue
                path, elapsed = _train_subset_expert(
                    subset, seed, args.data_dir, str(new_traj_dir),
                    n_steps=args.n_steps,
                )
                train_log.append({"task": name, "seed": seed,
                                   "traj_path": str(path),
                                   "elapsed_s": elapsed, "skipped": False})
                print(f"  [{i:>2}/{len(new_tasks)}] {name} seed={seed}  "
                      f"trained in {elapsed:.1f}s")
        total_train = time.time() - t0
        print(f"Training done in {total_train:.1f}s "
              f"({total_train / 60:.1f} min)")

    # Build combined registry
    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)

    cached_entries = _load_cached_entries(
        Path(args.cached_traj_dir), EXISTING_TASKS, SEEDS, probe,
    )
    print(f"\nCached entries loaded: {len(cached_entries)}")

    new_entries: list[FractalEntry] = []
    for name, subset in new_tasks.items():
        for seed in SEEDS:
            p = new_traj_dir / f"ext_{name}_seed{seed}_trajectory.npy"
            if not p.is_file():
                print(f"  MISSING  {p}")
                continue
            model = _load_model_from_traj(p)
            sig = _signature(model, probe)
            new_entries.append(FractalEntry(
                name=f"{name}_seed{seed}", signature=sig,
                metadata={"task": name, "seed": seed,
                          "trajectory_path": str(p),
                          "subset": list(subset),
                          "source": "sprint7_new"},
            ))
    print(f"New entries built: {len(new_entries)}")

    registry = FractalRegistry()
    for e in cached_entries + new_entries:
        registry.add(e)
    print(f"Total registry size: {len(registry)}")

    # ── Calibration ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CALIBRATION AT SCALE")
    print("=" * 72)
    cal = registry.calibrate_thresholds()
    cal_d = cal.to_dict()
    print(f"  n_tasks: {cal.n_tasks}")
    print(f"  within-task pairs: n={cal_d['within_distance_stats']['n']}, "
          f"min={cal_d['within_distance_stats']['min']:.3f}, "
          f"mean={cal_d['within_distance_stats']['mean']:.3f}, "
          f"max={cal_d['within_distance_stats']['max']:.3f}")
    print(f"  cross-task pairs:  n={cal_d['cross_distance_stats']['n']}, "
          f"min={cal_d['cross_distance_stats']['min']:.3f}, "
          f"mean={cal_d['cross_distance_stats']['mean']:.3f}, "
          f"max={cal_d['cross_distance_stats']['max']:.3f}")
    print(f"  match = {cal.match_threshold:.3f}  "
          f"spawn = {cal.spawn_threshold:.3f}  "
          f"gap = {cal.spawn_threshold - cal.match_threshold:.3f}")
    print(f"  overlap: {cal.overlap}")
    print(f"  (Sprint 6 baseline: MVP N=22 gave 5.72/7.56 gap=1.83;")
    print(f"                      full N=33 gave 7.14/7.59 gap=0.45)")

    # ── Leave-one-task-out retrieval across all 31 tasks ──────────
    print("\n" + "=" * 72)
    print("  LEAVE-ONE-TASK-OUT RETRIEVAL (all tasks)")
    print("=" * 72)
    all_tasks = EXISTING_TASKS + list(new_tasks.keys())
    rows: list[dict] = []
    t_query_total = 0.0
    for held_task in all_tasks:
        sub_registry = FractalRegistry()
        held_entries: list[FractalEntry] = []
        for e in registry.entries():
            if e.metadata.get("task") == held_task:
                held_entries.append(e)
            else:
                sub_registry.add(e)
        if not held_entries or len(sub_registry) == 0:
            continue
        for held in held_entries:
            t0 = time.time()
            res = sub_registry.find_nearest(held.signature, k=3,
                                              query_name=held.name)
            t_query_total += time.time() - t0
            near = res.nearest
            rows.append({
                "held_task": held_task,
                "held": held.name,
                "nearest": near.name if near else None,
                "nearest_task": (near.metadata.get("task")
                                  if near else None),
                "nearest_distance": res.distances[0] if res.distances else None,
                "second_task": (res.entries[1].metadata.get("task")
                                 if len(res.entries) >= 2 else None),
                "third_task": (res.entries[2].metadata.get("task")
                                if len(res.entries) >= 3 else None),
            })

    n_queries = len(rows)
    # Can we compute nearest-task correctness? For leave-one-task-out,
    # the held task is NOT in the sub-registry — so "correct" isn't
    # defined as "same task". We measure instead: does the nearest
    # route to a related task? (Qualitative; output recorded for
    # review.)
    # But we can compute: for each held task, is there internal
    # consistency — do the 3 held seed queries all route to the same
    # task? If yes, the signature is robust to seed variation.
    routing_consistency: dict[str, list[str]] = {}
    for r in rows:
        routing_consistency.setdefault(r["held_task"], []).append(
            r["nearest_task"] or "")
    n_consistent = sum(1 for ts in routing_consistency.values()
                       if len(set(ts)) == 1)
    n_tasks_tested = len(routing_consistency)

    avg_query_ms = (t_query_total / n_queries) * 1000 if n_queries else 0.0
    print(f"  {n_queries} leave-one-task-out queries across "
          f"{n_tasks_tested} tasks.")
    print(f"  Avg find_nearest(k=3) latency: {avg_query_ms:.2f} ms  "
          f"(N={len(registry)} registry)")
    print(f"  Per-task routing consistency (3 seeds → same nearest task): "
          f"{n_consistent}/{n_tasks_tested}")

    # For each held task: report what it routes to when held out
    print()
    print("  Held-task → dominant nearest task (3-seed vote):")
    for held_task in all_tasks:
        if held_task not in routing_consistency:
            continue
        near_tasks = routing_consistency[held_task]
        dominant = max(set(near_tasks), key=near_tasks.count)
        n_agree = sum(1 for t in near_tasks if t == dominant)
        note = "" if n_agree == len(near_tasks) else (
            f"  (split: {' / '.join(near_tasks)})")
        print(f"    {held_task:<22s} → {dominant}{note}")

    # ── Same-task retrieval check (sanity) ─────────────────────────
    # For each task, pick seed=2024, register seeds 42+101 of ALL
    # tasks, query. Does retrieval still find the same-task sibling?
    print()
    print("=" * 72)
    print("  SAME-TASK RETRIEVAL (all tasks, seed=2024 vs seeds 42+101)")
    print("=" * 72)
    held_out: list[FractalEntry] = []
    reg_42_101 = FractalRegistry()
    for e in registry.entries():
        if e.metadata.get("seed") == 2024:
            held_out.append(e)
        else:
            reg_42_101.add(e)
    n_correct = 0
    for h in held_out:
        res = reg_42_101.find_nearest(h.signature, k=1,
                                        query_name=h.name)
        near = res.nearest
        same = (near is not None
                and near.metadata.get("task") == h.metadata.get("task"))
        n_correct += int(same)
    acc_same_task = n_correct / max(len(held_out), 1)
    print(f"  Registry (seeds 42+101): {len(reg_42_101)} entries.")
    print(f"  Held-out (seed=2024): {len(held_out)} entries.")
    print(f"  Same-task top-1 retrieval: {n_correct}/{len(held_out)} "
          f"= {acc_same_task:.3f}")

    # ── Save results ──────────────────────────────────────────────
    out = {
        "n_tasks": len(all_tasks),
        "n_entries": len(registry),
        "new_tasks": {k: list(v) for k, v in new_tasks.items()},
        "training_log": train_log,
        "calibration": cal_d,
        "leave_one_task_out": {
            "n_queries": n_queries,
            "n_tasks_tested": n_tasks_tested,
            "n_routing_consistent": n_consistent,
            "avg_query_latency_ms": avg_query_ms,
            "rows": rows,
        },
        "same_task_retrieval": {
            "n_held_out": len(held_out),
            "n_correct": n_correct,
            "accuracy": acc_same_task,
        },
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
