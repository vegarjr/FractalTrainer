"""v3 Sprint 10 — Curated-registry experiment.

Sprint 9c inferred: compose wins when the candidate pool redundantly
covers the target's class-1 set. Sprint 9d showed that on a uniform
registry, compose can't be reliably routed-to beyond N<100. Sprint 10
tests the registry-design alternative: if we CURATE the registry for
high-overlap coverage in a specific region, does compose's niche
expand?

Anchor region: digits {0, 1, 2, 3, 4}.

Curated registry (15 tasks × 3 seeds = 45 entries), all class-1 sets
being subsets of {0, 1, 2, 3, 4}:
    All 10 of C(5, 3): anchor3_012, anchor3_013, anchor3_014,
                       anchor3_023, anchor3_024, anchor3_034,
                       anchor3_123, anchor3_124, anchor3_134,
                       anchor3_234.
    All 5  of C(5, 4): anchor4_0123, anchor4_0124, anchor4_0134,
                       anchor4_0234, anchor4_1234.

Uniform registry (20 tasks × 3 seeds = 60 entries): Sprint 7's
subset_* tasks (class-1 sets scattered across {0..9}, with low
overlap with anchor region by design).

Probe queries (5 tasks × 3 seeds = 15 entries trained): 5 randomly
chosen 2-subsets of the anchor region — queries whose class-1 sets
SHOULD have redundant coverage in the curated registry (each 2-subset
is contained in 3 of the curated 3-subsets + 3 of the 4-subsets), but
unlikely in the uniform one.

For each probe query q at each budget N ∈ {50, 100, 300, 1000, 5000}:
  acc_compose_curated = coverage-greedy K=3 from top-10 of Registry A
  acc_compose_uniform = coverage-greedy K=3 from top-10 of Registry B
  acc_spawn           = fresh MLP trained on N labels of q
  acc_top1_curated    = nearest-by-signature in Registry A
  acc_top1_uniform    = nearest-by-signature in Registry B

All evaluated on the same disjoint 1000-example MNIST test set.

Key question: is acc_compose_curated > acc_compose_uniform at ALL
budgets? Does the curated registry have a higher crossover budget
(where compose finally loses to spawn) than the uniform?
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
SEEDS = [42, 101, 2024]
ANCHOR = (0, 1, 2, 3, 4)

CURATED_3_SUBSETS = list(itertools.combinations(ANCHOR, 3))  # 10
CURATED_4_SUBSETS = list(itertools.combinations(ANCHOR, 4))  # 5
CURATED_TASKS: dict[str, tuple[int, ...]] = {}
for s in CURATED_3_SUBSETS:
    CURATED_TASKS[f"anchor3_{''.join(str(d) for d in s)}"] = tuple(s)
for s in CURATED_4_SUBSETS:
    CURATED_TASKS[f"anchor4_{''.join(str(d) for d in s)}"] = tuple(s)

# 5 of the C(5,2)=10 2-subsets — chosen to span the anchor region
PROBE_TASKS: dict[str, tuple[int, ...]] = {
    "probe2_01": (0, 1),
    "probe2_12": (1, 2),
    "probe2_23": (2, 3),
    "probe2_34": (3, 4),
    "probe2_04": (0, 4),
}

# Sprint 7's uniform registry
EXISTING_BINARY: dict[str, tuple[int, ...]] = {
    "parity":          (1, 3, 5, 7, 9),
    "high_vs_low":     (5, 6, 7, 8, 9),
    "primes_vs_rest":  (2, 3, 5, 7),
    "ones_vs_teens":   (0, 1, 2, 3, 4),
    "triangular":      (1, 3, 6),
    "fibonacci":       (1, 2, 3, 5, 8),
    "middle_456":      (4, 5, 6),
}


def _sprint7_new_tasks() -> dict[str, tuple[int, ...]]:
    existing = [frozenset(s) for s in EXISTING_BINARY.values()]

    def _is_novel(s: frozenset[int]) -> bool:
        full = frozenset(range(10))
        return all(s != ex and s != (full - ex) for ex in existing)

    cands = []
    for k in (3, 4, 5, 6):
        for c in itertools.combinations(range(10), k):
            if _is_novel(frozenset(c)):
                cands.append(tuple(sorted(c)))
    rng = random.Random(42)
    rng.shuffle(cands)
    return {
        "subset_" + "".join(str(d) for d in s): s
        for s in cands[:20]
    }


UNIFORM_TASKS = _sprint7_new_tasks()


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


class RelabeledMNIST(Dataset):
    def __init__(self, base, target: set[int]):
        self.base, self.target = base, set(target)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(int(y) in self.target)


def _train_expert(
    name: str, subset: tuple[int, ...], seed: int,
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
    ds = RelabeledMNIST(base, set(subset))
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
    run_id = f"ext_{name}_seed{seed}"
    trainer = InstrumentedTrainer(
        model=model, dataloader=loader, loss_fn=F.cross_entropy,
        hparams=hparams, snapshot_every=snapshot_every,
        out_dir=out_dir, run_id=run_id,
    )
    t0 = time.time()
    run = trainer.train(n_steps)
    elapsed = time.time() - t0
    return Path(run.snapshot_path), elapsed


def _make_budget_loader(subset: tuple[int, ...], N: int,
                         data_dir: str, seed: int) -> DataLoader:
    from torchvision import datasets, transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    ds = RelabeledMNIST(base, set(subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=N, replace=False)
    return DataLoader(Subset(ds, idx.tolist()),
                       batch_size=min(64, N), shuffle=False)


def _make_eval_loader(subset: tuple[int, ...], n_eval: int,
                       data_dir: str, seed: int) -> DataLoader:
    from torchvision import datasets, transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    ds = RelabeledMNIST(base, set(subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=n_eval, replace=False)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=64, shuffle=False)


def _compute_probs_and_labels(
    model: nn.Module, loader: DataLoader, n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)[:, :n_classes]
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
            labels_list.append(y.cpu().numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


def _greedy_coverage_select(
    cand_probs: list[np.ndarray], labels: np.ndarray, k: int,
) -> list[int]:
    n = len(cand_probs)
    if k >= n:
        return list(range(n))
    selected: list[int] = []
    remaining = set(range(n))
    running = np.zeros_like(cand_probs[0])
    for _ in range(k):
        best_acc, best_i = -1.0, None
        for i in remaining:
            trial = (running + cand_probs[i]) / (len(selected) + 1)
            acc = float((trial.argmax(axis=1) == labels).mean())
            if acc > best_acc:
                best_acc, best_i = acc, i
        assert best_i is not None
        selected.append(best_i)
        remaining.discard(best_i)
        running = running + cand_probs[best_i]
    return selected


def _ensemble_accuracy(probs_list: list[np.ndarray],
                        labels: np.ndarray) -> float:
    if not probs_list:
        return 0.0
    blend = np.mean(probs_list, axis=0)
    return float((blend.argmax(axis=1) == labels).mean())


def _spawn_with_budget(
    subset: tuple[int, ...], N: int, seed: int, data_dir: str,
    eval_loader: DataLoader, n_steps: int = 500,
) -> tuple[float, float]:
    budget_loader = _make_budget_loader(subset, N, data_dir, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    t0 = time.time()
    step = 0
    while step < n_steps:
        for x, y in budget_loader:
            if step >= n_steps: break
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1
    elapsed = time.time() - t0
    probs, labels = _compute_probs_and_labels(model, eval_loader, n_classes=2)
    acc = float((probs.argmax(axis=1) == labels).mean())
    return acc, elapsed


# ── Training phase ──

def train_all_new_experts(
    out_dir: Path, data_dir: str, n_steps: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    all_to_train = list(CURATED_TASKS.items()) + list(PROBE_TASKS.items())
    total = len(all_to_train) * len(SEEDS)
    print(f"Training {total} new experts "
          f"({len(CURATED_TASKS)} curated + {len(PROBE_TASKS)} probe)"
          f" × {len(SEEDS)} seeds.")
    i = 0
    t_start = time.time()
    for name, subset in all_to_train:
        for seed in SEEDS:
            i += 1
            p = out_dir / f"ext_{name}_seed{seed}_trajectory.npy"
            if p.is_file():
                print(f"  [{i:>2}/{total}] {name} seed={seed}  (cached)")
                log.append({"task": name, "seed": seed,
                             "traj_path": str(p), "elapsed_s": 0.0,
                             "skipped": True})
                continue
            path, elapsed = _train_expert(
                name, subset, seed, data_dir, str(out_dir),
                n_steps=n_steps,
            )
            log.append({"task": name, "seed": seed,
                         "traj_path": str(path),
                         "elapsed_s": elapsed, "skipped": False})
            print(f"  [{i:>2}/{total}] {name} seed={seed}  "
                  f"trained in {elapsed:.1f}s")
    print(f"Training wall clock: {time.time() - t_start:.1f}s")
    return log


def build_registry(
    task_dict: dict[str, tuple[int, ...]],
    traj_dir: Path, probe: torch.Tensor,
) -> tuple[FractalRegistry, dict[tuple[str, int], FractalEntry]]:
    reg = FractalRegistry()
    entries: dict[tuple[str, int], FractalEntry] = {}
    for name in task_dict:
        for seed in SEEDS:
            p = traj_dir / f"ext_{name}_seed{seed}_trajectory.npy"
            if not p.is_file():
                continue
            sig = _signature(_load_model(p), probe)
            e = FractalEntry(
                name=f"{name}_seed{seed}", signature=sig,
                metadata={"task": name, "seed": seed,
                          "trajectory_path": str(p)},
            )
            reg.add(e)
            entries[(name, seed)] = e
    return reg, entries


def run_budget_sweep_for_query(
    query_task: str, query_subset: tuple[int, ...],
    query_seed: int, query_entry: FractalEntry,
    registry: FractalRegistry, registry_label: str,
    budgets: list[int], data_dir: str, candidate_pool: int,
    k: int, budget_seed: int, spawn_seed: int,
    eval_loader: DataLoader, eval_labels: np.ndarray,
    candidate_probs_eval_cache: dict,
) -> list[dict]:
    res = registry.find_nearest(query_entry.signature,
                                  k=min(candidate_pool, len(registry)))
    cand_names = [e.name for e in res.entries]
    cand_models = [_load_model(Path(e.metadata["trajectory_path"]))
                   for e in res.entries]

    cand_probs_eval = []
    for i, m in enumerate(cand_models):
        key = (registry_label, query_task, cand_names[i])
        if key in candidate_probs_eval_cache:
            p = candidate_probs_eval_cache[key]
        else:
            p, _ = _compute_probs_and_labels(m, eval_loader, n_classes=2)
            candidate_probs_eval_cache[key] = p
        cand_probs_eval.append(p)

    rows: list[dict] = []
    acc_top1 = _ensemble_accuracy([cand_probs_eval[0]], eval_labels)
    for budget in budgets:
        budget_loader = _make_budget_loader(query_subset, budget,
                                              data_dir, budget_seed)
        cand_probs_budget = []
        budget_labels = None
        for m in cand_models:
            p, l = _compute_probs_and_labels(m, budget_loader, n_classes=2)
            cand_probs_budget.append(p)
            if budget_labels is None:
                budget_labels = l
        selected = _greedy_coverage_select(cand_probs_budget,
                                            budget_labels, k=k)
        acc_compose = _ensemble_accuracy(
            [cand_probs_eval[i] for i in selected], eval_labels)
        rows.append({
            "registry": registry_label,
            "query_task": query_task,
            "budget_N": budget,
            "acc_top1": acc_top1,
            "acc_compose": acc_compose,
            "compose_selected": [cand_names[i] for i in selected],
            "candidate_pool": cand_names,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-traj-dir", type=str,
                        default="results/sprint10_curated_trajectories")
    parser.add_argument("--sprint7-dir", type=str,
                        default="results/sprint7_v3_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--candidate-pool", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[50, 100, 300, 1000, 5000])
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=7777)
    parser.add_argument("--budget-seed", type=int, default=2024)
    parser.add_argument("--spawn-seed", type=int, default=2024)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--results-out", type=str,
                        default="results/curated_registry_experiment.json")
    args = parser.parse_args(argv)

    new_traj_dir = Path(args.new_traj_dir)
    training_log = []
    if not args.skip_training:
        training_log = train_all_new_experts(
            new_traj_dir, args.data_dir, args.n_steps)

    # Build probe batch (signature space)
    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)

    # Build curated registry (45 entries) — excludes probe tasks
    curated_reg, curated_entries = build_registry(
        CURATED_TASKS, new_traj_dir, probe)
    print(f"\nCurated registry (Registry A): "
          f"{len(curated_reg)} entries / {len(CURATED_TASKS)} tasks.")

    # Build probe entries (for signatures + oracle)
    _, probe_entries = build_registry(PROBE_TASKS, new_traj_dir, probe)
    print(f"Probe entries: {len(probe_entries)}")

    # Build uniform registry (Sprint 7 subset_* tasks from sprint7_dir)
    uniform_reg, _ = build_registry(
        UNIFORM_TASKS, Path(args.sprint7_dir), probe)
    print(f"Uniform registry (Registry B): "
          f"{len(uniform_reg)} entries / {len(UNIFORM_TASKS)} tasks.")

    # ── Run the sweep for each probe query ──
    all_rows: list[dict] = []
    spawn_rows: list[dict] = []
    candidate_probs_cache: dict = {}
    print()
    print("=" * 72)
    print("  PER-QUERY BUDGET SWEEP (Registry A vs B + spawn)")
    print("=" * 72)

    for probe_name, probe_subset in PROBE_TASKS.items():
        probe_entry = probe_entries.get((probe_name, 2024))
        if probe_entry is None:
            print(f"  MISSING probe_entry for {probe_name}")
            continue

        eval_loader = _make_eval_loader(
            probe_subset, args.n_eval, args.data_dir, args.eval_seed)
        _, eval_labels = _compute_probs_and_labels(
            MLP(), eval_loader, n_classes=2)  # just get labels (model noise)
        # Actually, compute labels without model forward:
        eval_labels_list = []
        for _, y in eval_loader:
            eval_labels_list.append(y.numpy())
        eval_labels = np.concatenate(eval_labels_list)

        # Registry A (curated)
        rows_A = run_budget_sweep_for_query(
            probe_name, probe_subset, 2024, probe_entry,
            curated_reg, "curated",
            args.budgets, args.data_dir, args.candidate_pool, args.k,
            args.budget_seed, args.spawn_seed, eval_loader, eval_labels,
            candidate_probs_cache,
        )
        all_rows.extend(rows_A)
        # Registry B (uniform)
        rows_B = run_budget_sweep_for_query(
            probe_name, probe_subset, 2024, probe_entry,
            uniform_reg, "uniform",
            args.budgets, args.data_dir, args.candidate_pool, args.k,
            args.budget_seed, args.spawn_seed, eval_loader, eval_labels,
            candidate_probs_cache,
        )
        all_rows.extend(rows_B)

        # Spawn (registry-independent — one per budget per query)
        for budget in args.budgets:
            acc_spawn, elapsed = _spawn_with_budget(
                probe_subset, budget, args.spawn_seed,
                args.data_dir, eval_loader, n_steps=args.n_steps,
            )
            spawn_rows.append({
                "query_task": probe_name, "budget_N": budget,
                "acc_spawn": acc_spawn, "elapsed_s": elapsed,
            })

        # Oracle: probe's own seed=42 expert
        oracle_entry = probe_entries.get((probe_name, 42))
        oracle_acc = None
        if oracle_entry:
            oracle_model = _load_model(
                Path(oracle_entry.metadata["trajectory_path"]))
            oracle_probs, _ = _compute_probs_and_labels(
                oracle_model, eval_loader, n_classes=2)
            oracle_acc = float((oracle_probs.argmax(axis=1)
                                 == eval_labels).mean())

        print(f"\n  query = {probe_name}  (y ∈ {set(probe_subset)})  "
              f"oracle={oracle_acc:.3f}")
        print(f"  {'N':>6s}  {'comp_A':>7s}  {'comp_B':>7s}  "
              f"{'spawn':>7s}  {'top1_A':>7s}  {'top1_B':>7s}")
        spawn_by_N = {r["budget_N"]: r["acc_spawn"]
                      for r in spawn_rows if r["query_task"] == probe_name}
        for budget in args.budgets:
            cA = next(r for r in rows_A if r["budget_N"] == budget)
            cB = next(r for r in rows_B if r["budget_N"] == budget)
            print(f"  {budget:>6d}  {cA['acc_compose']:>7.3f}  "
                  f"{cB['acc_compose']:>7.3f}  "
                  f"{spawn_by_N[budget]:>7.3f}  "
                  f"{cA['acc_top1']:>7.3f}  {cB['acc_top1']:>7.3f}")

    # ── Aggregate ──
    print()
    print("=" * 72)
    print("  AGGREGATE: compose_curated vs compose_uniform vs spawn")
    print("=" * 72)
    agg = {b: {"cA": [], "cB": [], "spawn": []} for b in args.budgets}
    for r in all_rows:
        agg[r["budget_N"]][
            "cA" if r["registry"] == "curated" else "cB"
        ].append(r["acc_compose"])
    for r in spawn_rows:
        agg[r["budget_N"]]["spawn"].append(r["acc_spawn"])

    print(f"  {'N':>6s}  "
          f"{'comp_A_mean':>12s}  {'comp_B_mean':>12s}  "
          f"{'spawn_mean':>11s}  {'ΔA−B':>7s}  {'ΔA−spawn':>9s}")
    summary = {}
    for budget in args.budgets:
        a = np.array(agg[budget]["cA"])
        b = np.array(agg[budget]["cB"])
        s = np.array(agg[budget]["spawn"])
        d_AB = float(a.mean() - b.mean())
        d_As = float(a.mean() - s.mean())
        summary[budget] = {
            "compose_curated_mean": float(a.mean()),
            "compose_uniform_mean": float(b.mean()),
            "spawn_mean": float(s.mean()),
            "delta_curated_minus_uniform": d_AB,
            "delta_curated_minus_spawn": d_As,
        }
        print(f"  {budget:>6d}  {a.mean():>12.4f}  {b.mean():>12.4f}  "
              f"{s.mean():>11.4f}  {d_AB:>+7.4f}  {d_As:>+9.4f}")

    # ── Verdict ──
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    # Does curated registry expand compose's niche?
    # A: does curated always beat uniform on compose?
    curated_beats_uniform_all = all(
        summary[b]["delta_curated_minus_uniform"] > 0
        for b in args.budgets)
    # B: at what budget does curated compose first LOSE to spawn?
    cross_A = next((b for b in args.budgets
                    if summary[b]["delta_curated_minus_spawn"] < 0), None)
    cross_B_bool = [summary[b]["compose_uniform_mean"]
                      < summary[b]["spawn_mean"]
                    for b in args.budgets]
    cross_B = next((args.budgets[i] for i, v in enumerate(cross_B_bool) if v),
                    None)

    if curated_beats_uniform_all:
        print("  Curated registry beats uniform on compose at ALL budgets.")
    else:
        fail_budgets = [b for b in args.budgets
                        if summary[b]["delta_curated_minus_uniform"] <= 0]
        print(f"  Curated loses or ties uniform at N ∈ {fail_budgets}.")
    print(f"  Curated compose first loses to spawn at: {cross_A}")
    print(f"  Uniform compose first loses to spawn at: {cross_B}")

    if cross_A is None:
        print("  → Curated compose NEVER loses to spawn in the sweep: "
              "strong registry-design effect.")
    elif cross_B is not None and cross_A > cross_B:
        print(f"  → Curated widens compose's niche ({cross_A} vs "
              f"{cross_B}): registry design matters.")
    elif cross_B is not None and cross_A == cross_B:
        print("  → No niche widening: curation doesn't expand compose.")
    else:
        print("  → Curated loses earlier than uniform: "
              "surprising, curation hurts.")

    out = {
        "anchor": list(ANCHOR),
        "curated_tasks": {k: list(v) for k, v in CURATED_TASKS.items()},
        "probe_tasks": {k: list(v) for k, v in PROBE_TASKS.items()},
        "uniform_tasks": {k: list(v) for k, v in UNIFORM_TASKS.items()},
        "all_rows": all_rows,
        "spawn_rows": spawn_rows,
        "per_budget_summary": {str(k): v for k, v in summary.items()},
        "verdict": {
            "curated_beats_uniform_at_all_budgets": curated_beats_uniform_all,
            "curated_first_loses_to_spawn_at_N": cross_A,
            "uniform_first_loses_to_spawn_at_N": cross_B,
        },
        "training_log": training_log,
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
