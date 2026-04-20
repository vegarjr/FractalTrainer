"""v3 Sprint 11 — Multi-region curated registry + content-aware router.

Sprint 10b showed that a single curated sub-registry (anchor
{0,1,2,3,4}) is sharp in-region but crashes OOD. Sprint 11 tests
the natural next step: a multi-region architecture with a
content-aware router.

Architecture:
    Region A sub-registry : curated around {0, 1, 2, 3, 4}  (Sprint 10)
    Region B sub-registry : curated around {5, 6, 7, 8, 9}  (new)
    Content-aware router  : given query target T,
                            - compute overlap(T, anchor_A) and overlap(T, anchor_B);
                            - if max overlap ≥ in_region_threshold: route to that
                              sub-registry and compose;
                            - else: fall through to spawn (cross-region).

Each sub-registry uses the same curated structure:
  All C(5, 3) + C(5, 4) subsets of its anchor = 15 tasks × 3 seeds.
  Region B = 15 new tasks × 3 seeds = 45 new experts to train.

Probe set (15 queries total):
    5 in-region-A   (Sprint 10's probe2_* 2-subsets of {0-4})
    5 in-region-B   (Sprint 10b's probe_oob_* 2-subsets of {5-9})
    5 cross-region  (spanning both anchors — {0,5}, {1,6}, {2,7},
                                               {3,8}, {4,9})

Expected pattern:
    in-region-A  → router picks Region A → compose at oracle accuracy
    in-region-B  → router picks Region B → compose at oracle accuracy
    cross-region → router rejects (both overlaps ≤ 0.5) → spawn wins

If this pattern holds, the Mixture-of-Fractals becomes a genuinely
general architecture: curated sub-registries cover known regions at
low compute, spawn covers cross-region / novel queries.
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

ANCHOR_A = (0, 1, 2, 3, 4)
ANCHOR_B = (5, 6, 7, 8, 9)


def _build_curated_tasks(anchor: tuple[int, ...], prefix: str
                          ) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for s in itertools.combinations(anchor, 3):
        out[f"{prefix}3_{''.join(str(d) for d in s)}"] = tuple(s)
    for s in itertools.combinations(anchor, 4):
        out[f"{prefix}4_{''.join(str(d) for d in s)}"] = tuple(s)
    return out


REGION_A_TASKS = _build_curated_tasks(ANCHOR_A, "anchor")  # prefix from S10
REGION_B_TASKS = _build_curated_tasks(ANCHOR_B, "anchorB")  # new

IN_A_PROBES = {
    "probe2_01": (0, 1), "probe2_12": (1, 2), "probe2_23": (2, 3),
    "probe2_34": (3, 4), "probe2_04": (0, 4),
}
IN_B_PROBES = {
    "probe_oob_56": (5, 6), "probe_oob_78": (7, 8),
    "probe_oob_59": (5, 9), "probe_oob_67": (6, 7),
    "probe_oob_89": (8, 9),
}
CROSS_PROBES = {
    "probe_cr_05": (0, 5), "probe_cr_16": (1, 6),
    "probe_cr_27": (2, 7), "probe_cr_38": (3, 8),
    "probe_cr_49": (4, 9),
}
ALL_PROBES: dict[str, tuple[int, ...]] = {}
ALL_PROBES.update(IN_A_PROBES)
ALL_PROBES.update(IN_B_PROBES)
ALL_PROBES.update(CROSS_PROBES)
PROBE_REGION: dict[str, str] = {}
for n in IN_A_PROBES: PROBE_REGION[n] = "in_A"
for n in IN_B_PROBES: PROBE_REGION[n] = "in_B"
for n in CROSS_PROBES: PROBE_REGION[n] = "cross"


# Sprint 7's uniform baseline
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
        if x.ndim > 2: x = x.view(x.size(0), -1)
        return self.net(x)


def _load_model(path: Path) -> nn.Module:
    t = np.load(path); final = t[-1]
    m = MLP(); offset = 0; sd = {}
    for shape, name in LAYER_SHAPES:
        size = int(np.prod(shape))
        sd[name] = torch.tensor(final[offset:offset + size].reshape(shape),
                                  dtype=torch.float32)
        offset += size
    m.load_state_dict(sd); m.eval()
    return m


def _mnist_probe(data_dir, n, seed):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(base), size=n, replace=False)
    return torch.stack([base[i][0] for i in idx.tolist()], dim=0)


def _signature(model, probe):
    with torch.no_grad():
        p = F.softmax(model(probe), dim=1)
    return p.flatten().cpu().numpy()


class RelabeledMNIST(Dataset):
    def __init__(self, base, target):
        self.base, self.target = base, set(target)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(int(y) in self.target)


def _train_expert(name, subset, seed, data_dir, out_dir,
                   n_steps=500, train_size=5000, batch_size=64):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    ds = RelabeledMNIST(base, set(subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=train_size, replace=False)
    loader = DataLoader(Subset(ds, idx.tolist()),
                         batch_size=batch_size, shuffle=True, drop_last=True)
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP()
    hparams = {"learning_rate": 0.01, "batch_size": batch_size,
               "weight_decay": 0.0, "dropout": 0.0,
               "init_seed": seed, "optimizer": "adam"}
    trainer = InstrumentedTrainer(
        model=m, dataloader=loader, loss_fn=F.cross_entropy,
        hparams=hparams, snapshot_every=10,
        out_dir=out_dir, run_id=f"ext_{name}_seed{seed}",
    )
    t0 = time.time()
    run = trainer.train(n_steps)
    return Path(run.snapshot_path), time.time() - t0


def _budget_loader(subset, N, data_dir, seed):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    ds = RelabeledMNIST(base, set(subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=N, replace=False)
    return DataLoader(Subset(ds, idx.tolist()),
                       batch_size=min(64, N), shuffle=False)


def _eval_loader(subset, n_eval, data_dir, seed):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    ds = RelabeledMNIST(base, set(subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=n_eval, replace=False)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=64, shuffle=False)


def _probs_and_labels(model, loader, n_classes):
    model.eval()
    pl, ll = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)[:, :n_classes]
            pl.append(F.softmax(logits, dim=1).cpu().numpy())
            ll.append(y.cpu().numpy())
    return np.concatenate(pl), np.concatenate(ll)


def _greedy_coverage(cand_probs, labels, k):
    n = len(cand_probs)
    if k >= n: return list(range(n))
    selected, rem = [], set(range(n))
    running = np.zeros_like(cand_probs[0])
    for _ in range(k):
        best_acc, best_i = -1.0, None
        for i in rem:
            trial = (running + cand_probs[i]) / (len(selected) + 1)
            acc = float((trial.argmax(axis=1) == labels).mean())
            if acc > best_acc: best_acc, best_i = acc, i
        selected.append(best_i); rem.discard(best_i)
        running = running + cand_probs[best_i]
    return selected


def _ensemble_acc(probs, labels):
    if not probs: return 0.0
    b = np.mean(probs, axis=0)
    return float((b.argmax(axis=1) == labels).mean())


def _spawn(subset, N, seed, data_dir, eval_loader, n_steps=500):
    bl = _budget_loader(subset, N, data_dir, seed)
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    step = 0
    while step < n_steps:
        for x, y in bl:
            if step >= n_steps: break
            opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
            step += 1
    p, l = _probs_and_labels(m, eval_loader, n_classes=2)
    return float((p.argmax(axis=1) == l).mean())


def _build_reg(task_dict, traj_dir, probe):
    reg = FractalRegistry()
    entries = {}
    for name in task_dict:
        for seed in SEEDS:
            p = traj_dir / f"ext_{name}_seed{seed}_trajectory.npy"
            if not p.is_file(): continue
            sig = _signature(_load_model(p), probe)
            e = FractalEntry(name=f"{name}_seed{seed}", signature=sig,
                              metadata={"task": name, "seed": seed,
                                        "trajectory_path": str(p)})
            reg.add(e); entries[(name, seed)] = e
    return reg, entries


def _anchor_overlap(target: frozenset[int], anchor: tuple[int, ...]
                     ) -> float:
    """Fraction of target's elements that are in anchor."""
    if not target: return 0.0
    a = frozenset(anchor)
    return len(target & a) / len(target)


def route_content_aware(
    target: frozenset[int], in_region_threshold: float = 0.75,
) -> tuple[str, dict[str, float]]:
    """Pick sub-registry by target-anchor overlap.
    Returns (decision, overlap_scores)."""
    overlaps = {
        "A": _anchor_overlap(target, ANCHOR_A),
        "B": _anchor_overlap(target, ANCHOR_B),
    }
    best = max(overlaps.values())
    if best >= in_region_threshold:
        return (max(overlaps, key=overlaps.get), overlaps)
    return ("spawn", overlaps)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", type=str,
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
    parser.add_argument("--in-region-threshold", type=float, default=0.75)
    parser.add_argument("--results-out", type=str,
                        default="results/multi_region_experiment.json")
    args = parser.parse_args(argv)

    curated_dir = Path(args.curated_dir)
    curated_dir.mkdir(parents=True, exist_ok=True)

    # ── Train missing experts (Region B + cross probes) ──
    train_needed = []
    for name, s in REGION_B_TASKS.items():
        for seed in SEEDS:
            if not (curated_dir / f"ext_{name}_seed{seed}_trajectory.npy").is_file():
                train_needed.append((name, s, seed))
    for name, s in CROSS_PROBES.items():
        for seed in SEEDS:
            if not (curated_dir / f"ext_{name}_seed{seed}_trajectory.npy").is_file():
                train_needed.append((name, s, seed))
    print(f"Need to train {len(train_needed)} new experts.")
    if train_needed:
        t0 = time.time()
        for i, (name, s, seed) in enumerate(train_needed, 1):
            _, elapsed = _train_expert(
                name, s, seed, args.data_dir, str(curated_dir))
            print(f"  [{i:>2}/{len(train_needed)}] {name} seed={seed}  "
                  f"trained in {elapsed:.1f}s")
        print(f"Training wall clock: {time.time() - t0:.1f}s")

    # ── Build registries ──
    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)
    reg_A, _ = _build_reg(REGION_A_TASKS, curated_dir, probe)
    reg_B, _ = _build_reg(REGION_B_TASKS, curated_dir, probe)
    reg_uniform, _ = _build_reg(UNIFORM_TASKS, Path(args.sprint7_dir), probe)
    _, probe_entries = _build_reg(ALL_PROBES, curated_dir, probe)
    print(f"\nRegion A sub-registry : {len(reg_A)} entries")
    print(f"Region B sub-registry : {len(reg_B)} entries")
    print(f"Uniform (Sprint 7)    : {len(reg_uniform)} entries")
    print(f"Probe entries         : {len(probe_entries)}")

    # ── Run sweep ──
    all_rows, spawn_rows, router_decisions = [], [], []
    print("\n" + "=" * 72)
    print("  MULTI-REGION EXPERIMENT")
    print("=" * 72)
    for probe_name, probe_subset in ALL_PROBES.items():
        q_entry = probe_entries.get((probe_name, 2024))
        if q_entry is None: continue

        target = frozenset(probe_subset)
        region_type = PROBE_REGION[probe_name]
        route_choice, overlaps = route_content_aware(
            target, args.in_region_threshold)
        router_decisions.append({
            "probe": probe_name, "target": sorted(target),
            "region_type": region_type, "overlaps": overlaps,
            "route": route_choice,
        })

        eval_loader = _eval_loader(
            probe_subset, args.n_eval, args.data_dir, args.eval_seed)
        ll = []
        for _, y in eval_loader: ll.append(y.numpy())
        eval_labels = np.concatenate(ll)

        # Oracle from seed=42
        oracle_entry = probe_entries.get((probe_name, 42))
        oracle_acc = None
        if oracle_entry:
            m = _load_model(Path(oracle_entry.metadata["trajectory_path"]))
            op, _ = _probs_and_labels(m, eval_loader, n_classes=2)
            oracle_acc = float((op.argmax(axis=1) == eval_labels).mean())

        # Compose on A, B, uniform (for every probe — for comparison)
        def compose_on(reg, reg_label):
            if len(reg) == 0: return None, None, None, None
            res = reg.find_nearest(q_entry.signature,
                                     k=min(args.candidate_pool, len(reg)))
            cand_names = [e.name for e in res.entries]
            cand_models = [_load_model(Path(e.metadata["trajectory_path"]))
                           for e in res.entries]
            cand_probs_eval = []
            for cm in cand_models:
                p, _ = _probs_and_labels(cm, eval_loader, n_classes=2)
                cand_probs_eval.append(p)
            top1_acc = _ensemble_acc([cand_probs_eval[0]], eval_labels)
            per_budget = {}
            for budget in args.budgets:
                bl = _budget_loader(probe_subset, budget, args.data_dir,
                                     args.budget_seed)
                cpb, bl_labels = [], None
                for cm in cand_models:
                    p, l = _probs_and_labels(cm, bl, n_classes=2)
                    cpb.append(p)
                    if bl_labels is None: bl_labels = l
                sel = _greedy_coverage(cpb, bl_labels, k=args.k)
                compose_acc = _ensemble_acc(
                    [cand_probs_eval[i] for i in sel], eval_labels)
                per_budget[budget] = {
                    "compose_acc": compose_acc,
                    "selected": [cand_names[i] for i in sel],
                }
            return top1_acc, per_budget, cand_names, None

        top1_A, pb_A, pool_A, _ = compose_on(reg_A, "A")
        top1_B, pb_B, pool_B, _ = compose_on(reg_B, "B")
        top1_U, pb_U, pool_U, _ = compose_on(reg_uniform, "uniform")

        # Spawn
        spawn_by_budget = {}
        for budget in args.budgets:
            s_acc = _spawn(probe_subset, budget, args.spawn_seed,
                             args.data_dir, eval_loader)
            spawn_by_budget[budget] = s_acc
            spawn_rows.append({"probe": probe_name, "budget": budget,
                                "acc_spawn": s_acc})

        # Per-probe rows
        for budget in args.budgets:
            # Router: chosen sub-registry or spawn
            if route_choice == "A":
                acc_routed = pb_A[budget]["compose_acc"]
                routed_pb = pb_A[budget]
            elif route_choice == "B":
                acc_routed = pb_B[budget]["compose_acc"]
                routed_pb = pb_B[budget]
            else:  # spawn
                acc_routed = spawn_by_budget[budget]
                routed_pb = None
            multi_region_oracle = max(pb_A[budget]["compose_acc"],
                                        pb_B[budget]["compose_acc"])
            all_rows.append({
                "probe": probe_name, "region_type": region_type,
                "target": sorted(target), "route": route_choice,
                "budget_N": budget, "oracle_acc": oracle_acc,
                "acc_compose_A": pb_A[budget]["compose_acc"],
                "acc_compose_B": pb_B[budget]["compose_acc"],
                "acc_compose_uniform": pb_U[budget]["compose_acc"],
                "acc_multi_region_oracle": multi_region_oracle,
                "acc_routed": acc_routed,
                "acc_spawn": spawn_by_budget[budget],
                "routed_selected": (routed_pb["selected"] if routed_pb
                                     else None),
            })

        print(f"\n  {probe_name:<17s} region={region_type:<5s} "
              f"target={sorted(target)}  overlaps A={overlaps['A']:.2f} "
              f"B={overlaps['B']:.2f}  → route={route_choice}")
        print(f"  {'N':>6s}  {'cA':>6s}  {'cB':>6s}  {'cU':>6s}  "
              f"{'multi':>6s}  {'routed':>6s}  {'spawn':>6s}  "
              f"{'oracle':>6s}")
        for budget in args.budgets:
            r = next(x for x in all_rows
                     if x["probe"] == probe_name and x["budget_N"] == budget)
            print(f"  {budget:>6d}  {r['acc_compose_A']:>6.3f}  "
                  f"{r['acc_compose_B']:>6.3f}  "
                  f"{r['acc_compose_uniform']:>6.3f}  "
                  f"{r['acc_multi_region_oracle']:>6.3f}  "
                  f"{r['acc_routed']:>6.3f}  "
                  f"{spawn_by_budget[budget]:>6.3f}  "
                  f"{oracle_acc:>6.3f}")

    # ── Aggregate by region_type ──
    print("\n" + "=" * 72)
    print("  AGGREGATE BY REGION TYPE")
    print("=" * 72)
    region_types = ["in_A", "in_B", "cross"]
    for rt in region_types:
        rt_rows = [r for r in all_rows if r["region_type"] == rt]
        n = len({r["probe"] for r in rt_rows})
        print(f"\n  region_type = {rt}  ({n} probes)")
        print(f"  {'N':>6s}  {'cA':>6s}  {'cB':>6s}  {'cU':>6s}  "
              f"{'multi':>6s}  {'routed':>6s}  {'spawn':>6s}")
        for budget in args.budgets:
            rb = [r for r in rt_rows if r["budget_N"] == budget]
            if not rb: continue
            cA = float(np.mean([r["acc_compose_A"] for r in rb]))
            cB = float(np.mean([r["acc_compose_B"] for r in rb]))
            cU = float(np.mean([r["acc_compose_uniform"] for r in rb]))
            mr = float(np.mean([r["acc_multi_region_oracle"] for r in rb]))
            rt_acc = float(np.mean([r["acc_routed"] for r in rb]))
            sp = float(np.mean([r["acc_spawn"] for r in rb]))
            print(f"  {budget:>6d}  {cA:>6.3f}  {cB:>6.3f}  {cU:>6.3f}  "
                  f"{mr:>6.3f}  {rt_acc:>6.3f}  {sp:>6.3f}")

    # ── Verdict ──
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    # In-region performance: does router correctly pick the right sub-registry?
    for rt in region_types:
        rows = [r for r in all_rows if r["region_type"] == rt]
        mean_routed = float(np.mean([r["acc_routed"] for r in rows]))
        mean_spawn = float(np.mean([r["acc_spawn"] for r in rows]))
        mean_uniform = float(np.mean([r["acc_compose_uniform"] for r in rows]))
        routes = {r["route"] for r in rows}
        print(f"  {rt:<6s}: routed mean={mean_routed:.3f}  "
              f"spawn mean={mean_spawn:.3f}  "
              f"uniform-compose={mean_uniform:.3f}  "
              f"(routes chosen: {routes})")

    out = {
        "anchor_A": list(ANCHOR_A),
        "anchor_B": list(ANCHOR_B),
        "in_region_threshold": args.in_region_threshold,
        "all_rows": all_rows,
        "router_decisions": router_decisions,
        "spawn_rows": spawn_rows,
    }
    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults saved: {args.results_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
