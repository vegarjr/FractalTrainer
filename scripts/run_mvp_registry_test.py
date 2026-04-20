"""v3 Sprint 3 MVP — populate FractalRegistry + run 3 retrieval tests.

Uses the 33 cached trained experts from the extended discriminability
experiment. Each expert's activation signature is computed on a fixed
100-image MNIST probe batch (same methodology as Option 2).

Three tests:
    A. Retrieval (leave-one-seed-out):
       Register all seed=42 and seed=101 models (22 entries). For each
       seed=2024 model, query the registry. Does find_nearest retrieve
       its matching task-sibling? Metric: top-1 accuracy.

    B. Generalization (leave-one-task-out):
       For each task T, register all tasks EXCEPT T (across all seeds).
       For each T-seed-2024 model, query the registry. What semantic
       neighbor does it pick? (Qualitative check.)

    C. Inference routing:
       For each seed=2024 model (the held-out), find the nearest entry
       in the registry (seeds 42+101 of the same task exist). Use that
       nearest entry's MODEL to predict on the task's test set. Compare
       to (a) random-pick baseline, (b) the oracle same-task expert.
       Answers: does routing by signature give you a useful expert?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402


# Same layout as discriminability_option2_activations.py
LAYER_SHAPES = [
    ((64, 784), "net.0.weight"),
    ((64,),     "net.0.bias"),
    ((32, 64),  "net.2.weight"),
    ((32,),     "net.2.bias"),
    ((10, 32),  "net.4.weight"),
    ((10,),     "net.4.bias"),
]


MNIST_TASKS = [
    "digit_class", "parity", "high_vs_low",
    "mod3", "mod5",
    "primes_vs_rest", "ones_vs_teens", "triangular",
    "fibonacci", "middle_456",
]
FASHION_TASKS = ["fashion_class"]
ALL_TASKS = MNIST_TASKS + FASHION_TASKS
SEEDS = [42, 101, 2024]


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
        state_dict[name] = torch.tensor(chunk.reshape(shape),
                                          dtype=torch.float32)
        offset += size
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _relabel(y: int, task: str) -> int:
    if task == "digit_class":
        return int(y)
    if task == "parity":
        return int(y % 2)
    if task == "high_vs_low":
        return int(y >= 5)
    if task == "mod3":
        return int(y % 3)
    if task == "mod5":
        return int(y % 5)
    if task == "primes_vs_rest":
        return int(y in (2, 3, 5, 7))
    if task == "ones_vs_teens":
        return int(y <= 4)
    if task == "triangular":
        return int(y in (1, 3, 6))
    if task == "fibonacci":
        return int(y in (1, 2, 3, 5, 8))
    if task == "middle_456":
        return int(y in (4, 5, 6))
    if task == "fashion_class":
        return int(y)
    raise ValueError(task)


class RelabeledDataset(Dataset):
    def __init__(self, base, task: str):
        self.base = base
        self.task = task

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, _relabel(int(y), self.task)


def _task_test_loader(task: str, data_dir: str, test_size: int, seed: int):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    if task == "fashion_class":
        base = datasets.FashionMNIST(
            str(Path(data_dir) / "FashionMNIST"),
            train=False, download=True, transform=transform)
        ds: Dataset = base
    else:
        base = datasets.MNIST(data_dir, train=False, download=True,
                              transform=transform)
        ds = RelabeledDataset(base, task)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=test_size, replace=False)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=64, shuffle=False)


def _evaluate(model: nn.Module, loader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _build_probe_batch(data_dir: str, n_probe: int,
                        probe_seed: int) -> torch.Tensor:
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=False, download=True,
                          transform=transform)
    rng = np.random.RandomState(probe_seed)
    idx = rng.choice(len(base), size=n_probe, replace=False)
    return torch.stack([base[i][0] for i in idx.tolist()], dim=0)


def _signature(model: nn.Module, probe: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(probe)
        probs = F.softmax(logits, dim=1)
    return probs.flatten().cpu().numpy()


# ── Test A: leave-one-seed-out retrieval ────────────────────────────

def _test_a_retrieval(
    registry: FractalRegistry,
    held_out: dict[tuple[str, int], FractalEntry],
) -> dict:
    print("\n" + "=" * 72)
    print("  TEST A — Leave-one-seed-out retrieval")
    print("=" * 72)
    print("  Registry contains seed=42 and seed=101 entries for each task.")
    print("  For each seed=2024 entry, find nearest and check task match.")
    print()

    correct = 0
    total = 0
    details: list[dict] = []
    for (task, seed), entry in held_out.items():
        res = registry.find_nearest(entry.signature, k=1,
                                     query_name=entry.name)
        near = res.nearest
        near_task = near.metadata.get("task") if near else None
        near_seed = near.metadata.get("seed") if near else None
        hit = near_task == task
        correct += int(hit)
        total += 1
        details.append({
            "held_out_task": task, "held_out_seed": seed,
            "nearest_name": near.name if near else None,
            "nearest_task": near_task,
            "nearest_seed": near_seed,
            "distance": res.distances[0] if res.distances else None,
            "correct": hit,
        })
        flag = "✓" if hit else "✗"
        print(f"  {flag}  {task:<16s} seed=2024 → {near.name if near else 'None'}  "
              f"(d={res.distances[0]:.3f})")

    acc = correct / max(total, 1)
    print()
    print(f"  Top-1 task accuracy: {correct}/{total} = {acc:.3f}")
    return {"top1_accuracy": acc, "correct": correct, "total": total,
            "details": details}


# ── Test B: leave-one-task-out semantic neighbor ────────────────────

def _test_b_leave_one_task(
    all_entries: dict[tuple[str, int], FractalEntry],
) -> dict:
    print("\n" + "=" * 72)
    print("  TEST B — Leave-one-task-out semantic nearest")
    print("=" * 72)
    print("  For each task T: register all tasks EXCEPT T. For each held-out")
    print("  T-seed-2024, find nearest. What task does it route to?")
    print()

    outcomes: list[dict] = []
    for held_task in ALL_TASKS:
        reg = FractalRegistry()
        # Register all entries except any belonging to held_task
        for (t, s), e in all_entries.items():
            if t != held_task:
                reg.add(e)
        held_entry = all_entries.get((held_task, 2024))
        if held_entry is None:
            continue
        res = reg.find_nearest(held_entry.signature, k=3,
                                query_name=f"heldout_{held_task}")
        ranked = [(e.metadata.get("task"), e.metadata.get("seed"), d)
                  for e, d in zip(res.entries, res.distances)]
        print(f"  held out: {held_task:<16s} → "
              f"nearest: {ranked[0][0]} (seed {ranked[0][1]}, d={ranked[0][2]:.3f})")
        if len(ranked) >= 2:
            print(f"                           2nd:      "
                  f"{ranked[1][0]} (seed {ranked[1][1]}, d={ranked[1][2]:.3f})")
        outcomes.append({
            "held_task": held_task,
            "top3": [{"task": t, "seed": s, "distance": float(d)}
                     for t, s, d in ranked],
        })
    return {"outcomes": outcomes}


# ── Test C: inference routing ───────────────────────────────────────

def _test_c_inference_routing(
    registry: FractalRegistry,
    held_out: dict[tuple[str, int], FractalEntry],
    traj_dir: Path,
    data_dir: str,
    test_size: int,
) -> dict:
    print("\n" + "=" * 72)
    print("  TEST C — Inference routing (routed vs random vs oracle)")
    print("=" * 72)
    print("  For each held-out seed=2024 model:")
    print("    routed   = use the registry-nearest expert's model")
    print("    oracle   = use the held-out model itself (upper bound)")
    print("    random   = use a random-registry expert (lower bound)")
    print("    Compare test accuracy on the held-out task's test set.")
    print()

    rng = np.random.RandomState(0)
    rows: list[dict] = []
    for (task, seed), entry in held_out.items():
        res = registry.find_nearest(entry.signature, k=1,
                                     query_name=entry.name)
        nearest_entry = res.nearest
        if nearest_entry is None:
            continue
        # Load the nearest-entry's model from its stored trajectory path
        routed_path = Path(nearest_entry.metadata["trajectory_path"])
        oracle_path = traj_dir / f"ext_{task}_seed{seed}_trajectory.npy"
        # Random: choose a random registered entry (not the nearest)
        other_entries = [e for e in registry.entries()
                         if e.name != nearest_entry.name]
        random_entry = other_entries[rng.randint(0, len(other_entries))]
        random_path = Path(random_entry.metadata["trajectory_path"])

        test_loader = _task_test_loader(task, data_dir, test_size,
                                         seed=7777)
        routed_model = _load_model(routed_path)
        oracle_model = _load_model(oracle_path)
        random_model = _load_model(random_path)

        acc_routed = _evaluate(routed_model, test_loader)
        acc_oracle = _evaluate(oracle_model, test_loader)
        acc_random = _evaluate(random_model, test_loader)

        rows.append({
            "task": task,
            "held_seed": seed,
            "nearest_name": nearest_entry.name,
            "nearest_task": nearest_entry.metadata.get("task"),
            "random_name": random_entry.name,
            "random_task": random_entry.metadata.get("task"),
            "acc_routed": acc_routed,
            "acc_oracle": acc_oracle,
            "acc_random": acc_random,
        })
        routed_vs_random = acc_routed - acc_random
        print(f"  {task:<16s}  routed={acc_routed:.3f}  "
              f"oracle={acc_oracle:.3f}  random={acc_random:.3f}  "
              f"(Δ routed−random = {routed_vs_random:+.3f})")

    if not rows:
        return {"rows": [], "summary": {}}

    mean_routed = float(np.mean([r["acc_routed"] for r in rows]))
    mean_oracle = float(np.mean([r["acc_oracle"] for r in rows]))
    mean_random = float(np.mean([r["acc_random"] for r in rows]))

    print()
    print(f"  Mean routed accuracy: {mean_routed:.4f}")
    print(f"  Mean oracle accuracy: {mean_oracle:.4f}  (upper bound)")
    print(f"  Mean random accuracy: {mean_random:.4f}  (lower bound)")
    print(f"  Routing lift over random: +{mean_routed - mean_random:.4f}")
    print(f"  Routing gap below oracle: -{mean_oracle - mean_routed:.4f}")

    return {
        "rows": rows,
        "summary": {
            "mean_routed": mean_routed,
            "mean_oracle": mean_oracle,
            "mean_random": mean_random,
            "lift_over_random": mean_routed - mean_random,
            "gap_below_oracle": mean_oracle - mean_routed,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--registry-out", type=str,
                        default="results/fractal_registry_mvp.json")
    parser.add_argument("--results-out", type=str,
                        default="results/mvp_registry_test.json")
    args = parser.parse_args(argv)

    traj_dir = Path(args.traj_dir)
    probe = _build_probe_batch(args.data_dir, args.n_probe, args.probe_seed)
    print(f"Probe batch: {args.n_probe} MNIST test images (probe_seed={args.probe_seed})")

    # Build all 33 entries
    all_entries: dict[tuple[str, int], FractalEntry] = {}
    print(f"\nBuilding {len(ALL_TASKS) * len(SEEDS)} entries...")
    for task in ALL_TASKS:
        for seed in SEEDS:
            traj_path = traj_dir / f"ext_{task}_seed{seed}_trajectory.npy"
            if not traj_path.is_file():
                print(f"  SKIP {task} seed={seed} (trajectory missing)")
                continue
            model = _load_model(traj_path)
            sig = _signature(model, probe)
            entry = FractalEntry(
                name=f"{task}_seed{seed}",
                signature=sig,
                metadata={
                    "task": task, "seed": seed,
                    "trajectory_path": str(traj_path),
                    "signature_dim": int(sig.size),
                },
            )
            all_entries[(task, seed)] = entry
    print(f"Built {len(all_entries)} entries, signature_dim={probe.shape[0] * 10}")

    # Registry for test A + C: seed 42 + 101 (22 entries)
    registry = FractalRegistry()
    held_out: dict[tuple[str, int], FractalEntry] = {}
    for (t, s), e in all_entries.items():
        if s == 2024:
            held_out[(t, s)] = e
        else:
            registry.add(e)
    print(f"Test A/C registry: {len(registry)} entries  (held out: "
          f"{len(held_out)} seed=2024 entries)")

    result_a = _test_a_retrieval(registry, held_out)
    result_b = _test_b_leave_one_task(all_entries)
    result_c = _test_c_inference_routing(
        registry, held_out, traj_dir, args.data_dir, args.test_size,
    )

    # Save the registry (test A/C version)
    registry.save(args.registry_out)
    print(f"\nRegistry saved: {args.registry_out}")

    # Save full results
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n_probe": args.n_probe,
            "probe_seed": args.probe_seed,
            "n_entries": len(all_entries),
            "test_a_retrieval": result_a,
            "test_b_leave_one_task": result_b,
            "test_c_inference_routing": result_c,
        }, f, indent=2, default=str)
    print(f"results saved: {out_path}")

    # Final verdict
    print()
    print("=" * 72)
    print("  MVP VERDICT")
    print("=" * 72)
    a = result_a["top1_accuracy"]
    c_lift = result_c["summary"].get("lift_over_random", float("nan"))
    c_gap = result_c["summary"].get("gap_below_oracle", float("nan"))
    lines: list[str] = []
    if a >= 0.9:
        lines.append(f"Test A (retrieval): PASS — {a:.3f} top-1 accuracy.")
        lines.append("  → Routing correctly identifies same-task experts.")
    elif a >= 0.7:
        lines.append(f"Test A (retrieval): WEAK — {a:.3f} top-1.")
    else:
        lines.append(f"Test A (retrieval): FAIL — {a:.3f} top-1.")
    if c_lift > 0.1:
        lines.append(f"Test C (routing): PASS — "
                      f"routed +{c_lift:.3f} over random, "
                      f"-{c_gap:.3f} below oracle.")
        lines.append("  → Signature routing delivers real utility.")
    elif c_lift > 0:
        lines.append(f"Test C (routing): MARGINAL — +{c_lift:.3f} over random.")
    else:
        lines.append(f"Test C (routing): FAIL — no lift over random.")
    for line in lines:
        print(f"  {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
