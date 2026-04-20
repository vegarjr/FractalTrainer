"""v3 Sprint 8 — Flat vs Hierarchical routing comparison.

Demonstrates the two-level (dataset-coarse + task-fine) routing on a
mixed-dataset registry: 30 MNIST tasks + 11 Fashion-MNIST tasks.

Central question: at 41 tasks / 123 entries, does hierarchical routing
(a) match flat routing on same-dataset queries, (b) avoid cross-dataset
contamination, (c) provide measurable latency benefit?

Registry composition (post-build):
  MNIST kingdom:   30 tasks × 3 seeds = 90 entries
    - existing: digit_class, parity, high_vs_low, mod3, mod5,
      primes_vs_rest, ones_vs_teens, triangular, fibonacci, middle_456
    - Sprint 7 additions: 20 subset_* binary tasks
  Fashion kingdom: 11 tasks × 3 seeds = 33 entries
    - existing: fashion_class (10-class)
    - Sprint 8 additions: 10 binary fashion_* tasks
  Total: 41 tasks / 123 entries.

Tests:
  A. Same-dataset retrieval accuracy — leave one seed out per task,
     query against seeds 42+101, does hierarchical pick the same
     nearest-task as flat?
  B. Cross-dataset contamination — when Fashion seed=2024 queries
     against a registry with both MNIST + Fashion entries, does the
     FLAT nearest ever land in the MNIST kingdom? (Hierarchical is
     guaranteed not to, by construction, if the centroid is right.)
  C. Latency — flat O(N) scan vs hierarchical (O(datasets) centroid
     + O(N_dataset) fine search) at N=123.
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


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.registry import (  # noqa: E402
    FractalEntry,
    FractalRegistry,
    HierarchicalRegistry,
)


LAYER_SHAPES = [
    ((64, 784), "net.0.weight"), ((64,), "net.0.bias"),
    ((32, 64),  "net.2.weight"), ((32,), "net.2.bias"),
    ((10, 32),  "net.4.weight"), ((10,), "net.4.bias"),
]

SEEDS = [42, 101, 2024]

# MNIST tasks from Sprints 3-7
MNIST_CACHED_TASKS = [
    "digit_class", "parity", "high_vs_low", "mod3", "mod5",
    "primes_vs_rest", "ones_vs_teens", "triangular",
    "fibonacci", "middle_456",
]
EXISTING_BINARY_SUBSETS = [
    frozenset({1, 3, 5, 7, 9}),
    frozenset({5, 6, 7, 8, 9}),
    frozenset({2, 3, 5, 7}),
    frozenset({0, 1, 2, 3, 4}),
    frozenset({1, 3, 6}),
    frozenset({1, 2, 3, 5, 8}),
    frozenset({4, 5, 6}),
]

# Fashion tasks for Sprint 8
FASHION_BINARY_TASKS: dict[str, tuple[int, ...]] = {
    "fashion_upperbody":  (0, 2, 3, 4, 6),
    "fashion_footwear":   (5, 7, 9),
    "fashion_even_idx":   (0, 2, 4, 6, 8),
    "fashion_first_half": (0, 1, 2, 3, 4),
    "fashion_corners":    (0, 1, 8, 9),
    "fashion_middle":     (3, 4, 5, 6),
    "fashion_no_bag":     (0, 1, 2, 3, 4, 6, 7, 9),
    "fashion_warm":       (2, 3, 4, 6),
    "fashion_athletic":   (1, 5, 7),
    "fashion_casual":     (0, 5, 7),
}


def _sprint7_new_tasks() -> dict[str, tuple[int, ...]]:
    """Reproduce Sprint 7 task sampler."""
    def _is_novel(s: frozenset[int]) -> bool:
        full = frozenset(range(10))
        return all(s != ex and s != (full - ex) for ex in EXISTING_BINARY_SUBSETS)

    candidates = []
    for k in (3, 4, 5, 6):
        for c in itertools.combinations(range(10), k):
            if _is_novel(frozenset(c)):
                candidates.append(tuple(sorted(c)))
    rng = random.Random(42)
    rng.shuffle(candidates)
    return {
        "subset_" + "".join(str(d) for d in s): s
        for s in candidates[:20]
    }


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


def _load_model(traj_path: Path) -> nn.Module:
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


def _mnist_probe(data_dir: str, n: int, seed: int) -> torch.Tensor:
    """MNIST-based probe batch. Used for ALL signatures (including
    Fashion experts) so all signatures live in the same space."""
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


def _build_entry(traj_path: Path, task: str, seed: int, dataset: str,
                 probe: torch.Tensor) -> FractalEntry:
    model = _load_model(traj_path)
    sig = _signature(model, probe)
    return FractalEntry(
        name=f"{task}_seed{seed}",
        signature=sig,
        metadata={
            "task": task, "seed": seed, "dataset": dataset,
            "trajectory_path": str(traj_path),
        },
    )


def build_combined_registry(
    mnist_cached_dir: Path, sprint7_dir: Path, sprint8_dir: Path,
    probe: torch.Tensor,
) -> tuple[FractalRegistry, HierarchicalRegistry, list[FractalEntry]]:
    flat = FractalRegistry()
    hier = HierarchicalRegistry(dataset_key="dataset")
    all_entries: list[FractalEntry] = []

    # MNIST kingdom — cached tasks (Sprint 3/4/6)
    for task in MNIST_CACHED_TASKS:
        for seed in SEEDS:
            p = mnist_cached_dir / f"ext_{task}_seed{seed}_trajectory.npy"
            if not p.is_file():
                continue
            e = _build_entry(p, task, seed, "mnist", probe)
            all_entries.append(e)

    # MNIST kingdom — Sprint 7 subset tasks
    for task in _sprint7_new_tasks():
        for seed in SEEDS:
            p = sprint7_dir / f"ext_{task}_seed{seed}_trajectory.npy"
            if not p.is_file():
                continue
            e = _build_entry(p, task, seed, "mnist", probe)
            all_entries.append(e)

    # Fashion kingdom — cached fashion_class (one 10-class task) lives
    # in the same dir as the MNIST cached tasks.
    for seed in SEEDS:
        p = mnist_cached_dir / f"ext_fashion_class_seed{seed}_trajectory.npy"
        if not p.is_file():
            continue
        e = _build_entry(p, "fashion_class", seed, "fashion", probe)
        all_entries.append(e)

    # Fashion kingdom — Sprint 8 binary subsets
    for task in FASHION_BINARY_TASKS:
        for seed in SEEDS:
            p = sprint8_dir / f"ext_{task}_seed{seed}_trajectory.npy"
            if not p.is_file():
                print(f"  MISSING  {p.name}")
                continue
            e = _build_entry(p, task, seed, "fashion", probe)
            all_entries.append(e)

    for e in all_entries:
        flat.add(e)
        hier.add(e)
    return flat, hier, all_entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-cached-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--sprint7-dir", type=str,
                        default="results/sprint7_v3_trajectories")
    parser.add_argument("--sprint8-dir", type=str,
                        default="results/sprint8_v3_fashion_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--results-out", type=str,
                        default="results/sprint8_v3_hierarchical_demo.json")
    args = parser.parse_args(argv)

    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)
    flat, hier, all_entries = build_combined_registry(
        Path(args.mnist_cached_dir), Path(args.sprint7_dir),
        Path(args.sprint8_dir), probe,
    )

    print(f"Built registry: {len(all_entries)} entries "
          f"({len(flat)} in flat view, {len(hier)} in hierarchical).")
    print(f"Datasets: {hier.datasets()}")
    for ds in hier.datasets():
        sub = hier.sub_registry(ds)
        tasks = sorted({e.metadata['task'] for e in sub.entries()})
        print(f"  {ds:<10s} {len(sub):>4d} entries / "
              f"{len(tasks)} tasks")

    # Centroid distance matrix (sanity)
    centroids = {ds: hier.centroid(ds) for ds in hier.datasets()}
    if len(centroids) >= 2:
        datasets_list = list(centroids.keys())
        print(f"\nInter-centroid distances:")
        for i, d1 in enumerate(datasets_list):
            for d2 in datasets_list[i + 1:]:
                d = float(np.linalg.norm(centroids[d1] - centroids[d2]))
                print(f"  {d1} ↔ {d2}: {d:.3f}")

    # ── Test A: same-task retrieval, flat vs hierarchical ─────────
    # Leave out seed=2024, register seeds 42+101 of all tasks.
    # For each held-out entry: does flat (or hierarchical) find the
    # same-task sibling as top-1?
    print("\n" + "=" * 72)
    print("  TEST A — Same-task retrieval (flat vs hierarchical)")
    print("=" * 72)
    flat_AB = FractalRegistry()
    hier_AB = HierarchicalRegistry(dataset_key="dataset")
    held_out: list[FractalEntry] = []
    for e in all_entries:
        if e.metadata["seed"] == 2024:
            held_out.append(e)
        else:
            flat_AB.add(e)
            hier_AB.add(e)
    print(f"  Registry (seeds 42+101): {len(flat_AB)} entries.")
    print(f"  Held out (seed=2024): {len(held_out)} entries.")

    flat_correct = 0
    hier_correct = 0
    hier_dataset_correct = 0
    disagreements: list[dict] = []
    for h in held_out:
        # Flat
        flat_res = flat_AB.find_nearest(h.signature, k=1)
        flat_near = flat_res.nearest
        flat_same = (flat_near is not None
                     and flat_near.metadata.get("task")
                         == h.metadata.get("task"))
        flat_correct += int(flat_same)

        # Hierarchical
        hier_dec = hier_AB.find_nearest_hierarchical(h.signature, k=1)
        hier_near = hier_dec.retrieval.nearest
        hier_same = (hier_near is not None
                     and hier_near.metadata.get("task")
                         == h.metadata.get("task"))
        hier_correct += int(hier_same)
        hier_dataset_same = (hier_dec.chosen_dataset
                             == h.metadata.get("dataset"))
        hier_dataset_correct += int(hier_dataset_same)

        if flat_same != hier_same or not hier_dataset_same:
            disagreements.append({
                "held": h.name, "held_dataset": h.metadata["dataset"],
                "flat_nearest": flat_near.name if flat_near else None,
                "flat_same_task": flat_same,
                "hier_dataset": hier_dec.chosen_dataset,
                "hier_dataset_correct": hier_dataset_same,
                "hier_nearest": hier_near.name if hier_near else None,
                "hier_same_task": hier_same,
                "all_centroid_distances": hier_dec.all_centroid_distances,
            })

    n = len(held_out)
    print(f"  Flat         same-task@1: {flat_correct}/{n} = "
          f"{flat_correct/n:.3f}")
    print(f"  Hierarchical same-task@1: {hier_correct}/{n} = "
          f"{hier_correct/n:.3f}")
    print(f"  Hierarchical coarse-gate dataset-correct: "
          f"{hier_dataset_correct}/{n} = {hier_dataset_correct/n:.3f}")
    if disagreements:
        print(f"  {len(disagreements)} disagreement(s):")
        for d in disagreements[:10]:
            print(f"    {d['held']:<30s}  flat→{d['flat_nearest']:<25s}"
                  f"  hier→{d['hier_nearest'] if d['hier_nearest'] else 'None'} "
                  f"(dataset={d['hier_dataset']}, "
                  f"correct-ds={d['hier_dataset_correct']})")

    # ── Test B: cross-dataset contamination ──────────────────────
    # For each held-out Fashion entry, does flat routing's top-1
    # ever land in the MNIST kingdom?
    print("\n" + "=" * 72)
    print("  TEST B — Cross-dataset contamination under flat routing")
    print("=" * 72)
    contamination_rows: list[dict] = []
    for h in held_out:
        flat_res = flat_AB.find_nearest(h.signature, k=3)
        held_ds = h.metadata["dataset"]
        top1_ds = (flat_res.nearest.metadata.get("dataset")
                   if flat_res.nearest else None)
        top3_ds = [e.metadata.get("dataset") for e in flat_res.entries]
        n_cross_in_top3 = sum(1 for ds in top3_ds if ds != held_ds)
        contamination_rows.append({
            "held": h.name, "held_dataset": held_ds,
            "flat_top1_dataset": top1_ds,
            "flat_top1_same_dataset": top1_ds == held_ds,
            "flat_top3_datasets": top3_ds,
            "n_cross_dataset_in_top3": n_cross_in_top3,
        })
    n_top1_same = sum(r["flat_top1_same_dataset"]
                      for r in contamination_rows)
    n_any_cross_top3 = sum(r["n_cross_dataset_in_top3"] > 0
                            for r in contamination_rows)
    print(f"  Flat top-1 same-dataset: {n_top1_same}/{n} = "
          f"{n_top1_same / n:.3f}")
    print(f"  Flat top-3 has ≥1 cross-dataset entry: "
          f"{n_any_cross_top3}/{n} = {n_any_cross_top3 / n:.3f}")
    # By construction, hierarchical routing never crosses datasets
    # after the coarse gate. Contamination rate is exactly the
    # coarse-gate miss rate, which was reported in Test A.

    # ── Test C: latency ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TEST C — Latency (flat O(N) vs hierarchical)")
    print("=" * 72)
    # Warm up
    for _ in range(3):
        flat_AB.find_nearest(held_out[0].signature, k=1)
        hier_AB.find_nearest_hierarchical(held_out[0].signature, k=1)
    n_trials = 100
    t0 = time.time()
    for _ in range(n_trials):
        for h in held_out:
            flat_AB.find_nearest(h.signature, k=1)
    flat_total = time.time() - t0
    flat_per_query = flat_total / (n_trials * n)
    t0 = time.time()
    for _ in range(n_trials):
        for h in held_out:
            hier_AB.find_nearest_hierarchical(h.signature, k=1)
    hier_total = time.time() - t0
    hier_per_query = hier_total / (n_trials * n)

    print(f"  Flat:         {flat_per_query * 1000:.3f} ms/query "
          f"(N={len(flat_AB)})")
    print(f"  Hierarchical: {hier_per_query * 1000:.3f} ms/query "
          f"(N={len(hier_AB)}, 2 datasets)")
    print(f"  Ratio: {flat_per_query / hier_per_query:.2f}× "
          f"{'faster' if hier_per_query < flat_per_query else 'slower'} "
          f"for hierarchical")

    # ── Verdict ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    if hier_correct >= flat_correct and hier_dataset_correct == n:
        verdict_a = "PASS"
        reason_a = (f"hierarchical matches or beats flat same-task@1 "
                    f"({hier_correct}/{n} vs {flat_correct}/{n}) and "
                    f"coarse gate is perfect ({n}/{n}).")
    elif hier_dataset_correct < n:
        verdict_a = "COARSE GATE LEAKS"
        reason_a = (f"coarse gate routes to wrong dataset "
                    f"{n - hier_dataset_correct}/{n} times — contamination "
                    "possible.")
    else:
        verdict_a = "HIERARCHICAL LOSES SOME"
        reason_a = (f"hierarchical same-task {hier_correct}/{n} < "
                    f"flat {flat_correct}/{n} despite perfect coarse "
                    "gate — sub-registry task-level miss.")
    print(f"  A. Retrieval: {verdict_a} — {reason_a}")

    flat_contamination_rate = 1 - n_top1_same / n
    if flat_contamination_rate == 0:
        verdict_b = "NO CONTAMINATION"
        reason_b = ("even flat routing lands same-dataset 100% — "
                    "at this scale, hierarchical adds no contamination "
                    "protection that flat wasn't already providing.")
    elif flat_contamination_rate < 0.05:
        verdict_b = "RARE CONTAMINATION"
        reason_b = (f"flat crosses datasets {flat_contamination_rate:.1%} "
                    "of the time — hierarchical's coarse gate prevents it.")
    else:
        verdict_b = "MEANINGFUL CONTAMINATION"
        reason_b = (f"flat crosses datasets {flat_contamination_rate:.1%} "
                    "of the time — hierarchical fixes this by construction.")
    print(f"  B. Contamination: {verdict_b} — {reason_b}")

    if hier_per_query < flat_per_query:
        verdict_c = f"FASTER ({flat_per_query / hier_per_query:.2f}×)"
    elif hier_per_query > flat_per_query * 1.2:
        verdict_c = f"SLOWER ({hier_per_query / flat_per_query:.2f}×)"
    else:
        verdict_c = "INDISTINGUISHABLE"
    print(f"  C. Latency: {verdict_c} at N={len(flat_AB)}")

    out = {
        "n_entries": len(all_entries),
        "datasets": hier.datasets(),
        "test_a": {
            "n_queries": n,
            "flat_same_task_top1": flat_correct,
            "hier_same_task_top1": hier_correct,
            "hier_dataset_correct": hier_dataset_correct,
            "verdict": verdict_a,
            "reason": reason_a,
            "disagreements": disagreements,
        },
        "test_b": {
            "n_queries": n,
            "flat_top1_same_dataset": n_top1_same,
            "flat_top3_any_cross_dataset": n_any_cross_top3,
            "contamination_rate": flat_contamination_rate,
            "verdict": verdict_b,
            "reason": reason_b,
            "rows": contamination_rows,
        },
        "test_c": {
            "n_trials": n_trials,
            "flat_ms_per_query": flat_per_query * 1000,
            "hier_ms_per_query": hier_per_query * 1000,
            "verdict": verdict_c,
        },
        "inter_centroid_distances": {
            f"{d1}|{d2}": float(np.linalg.norm(centroids[d1] - centroids[d2]))
            for i, d1 in enumerate(hier.datasets())
            for d2 in hier.datasets()[i + 1:]
        } if len(centroids) >= 2 else {},
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
