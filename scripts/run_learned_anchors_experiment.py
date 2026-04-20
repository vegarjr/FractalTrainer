"""v3 Sprint 12 — Learned anchors via Jaccard clustering.

Sprint 11 hand-picked the anchor split {0-4} / {5-9}. This was
another form of tweaking-in-my-favor: how did I know those were
the right regions? Sprint 12 removes that dependency by
**discovering** anchor regions from the registered experts'
label-set structure.

Method:
  1. Pool 50 registered task names (no probes):
     - 30 Sprint 10+11 curated tasks (anchor*, anchorB*)
     - 20 Sprint 7 uniform tasks (subset_*)
  2. Compute pairwise Jaccard distance between every pair.
  3. Agglomerative cluster with average linkage.
  4. Cut at k = 2, 3, 4.
  5. For each cluster, learned anchor = union of cluster members'
     class-1 sets.
  6. Route Sprint 11's 15 probes through learned anchors:
     - overlap(target, learned_anchor) ≥ threshold → route to
       that cluster's sub-registry
     - else → fall through to spawn.
  7. Compose on chosen sub-registry and measure accuracy on the
     same disjoint 1000-example test set used in Sprint 11.
  8. Compare learned-anchor routing to Sprint 11's hand-picked
     routing on every probe.

Success criteria:
  A. At k=2, clusters should approximately reproduce the {0-4} /
     {5-9} split (i.e. the anchor3/anchor4 vs anchorB3/anchorB4
     distinction).
  B. Routing decisions on the 15 probes should match Sprint 11
     (in_A → cluster A, in_B → cluster B, cross → fall through).
  C. Compose accuracies should match Sprint 11's numbers within
     noise.

If all three hold, the Mixture-of-Fractals architecture is
self-organizing: feed in experts, the regions emerge from data.
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

from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402


LAYER_SHAPES = [
    ((64, 784), "net.0.weight"), ((64,), "net.0.bias"),
    ((32, 64),  "net.2.weight"), ((32,), "net.2.bias"),
    ((10, 32),  "net.4.weight"), ((10,), "net.4.bias"),
]
SEEDS = [42, 101, 2024]


# ── Task definitions (all pool members) ──
def _build_curated_tasks(anchor, prefix):
    out = {}
    for s in itertools.combinations(anchor, 3):
        out[f"{prefix}3_{''.join(str(d) for d in s)}"] = tuple(s)
    for s in itertools.combinations(anchor, 4):
        out[f"{prefix}4_{''.join(str(d) for d in s)}"] = tuple(s)
    return out


CURATED_A = _build_curated_tasks((0, 1, 2, 3, 4), "anchor")
CURATED_B = _build_curated_tasks((5, 6, 7, 8, 9), "anchorB")


EXISTING_BINARY = {
    "parity":          (1, 3, 5, 7, 9),
    "high_vs_low":     (5, 6, 7, 8, 9),
    "primes_vs_rest":  (2, 3, 5, 7),
    "ones_vs_teens":   (0, 1, 2, 3, 4),
    "triangular":      (1, 3, 6),
    "fibonacci":       (1, 2, 3, 5, 8),
    "middle_456":      (4, 5, 6),
}


def _sprint7_new_tasks():
    existing = [frozenset(s) for s in EXISTING_BINARY.values()]

    def _is_novel(s):
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


# Probes (same as Sprint 11)
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
ALL_PROBES = {**IN_A_PROBES, **IN_B_PROBES, **CROSS_PROBES}
PROBE_REGION = {}
for n in IN_A_PROBES: PROBE_REGION[n] = "in_A"
for n in IN_B_PROBES: PROBE_REGION[n] = "in_B"
for n in CROSS_PROBES: PROBE_REGION[n] = "cross"


# ── Jaccard clustering ──
def _jaccard(a: frozenset, b: frozenset) -> float:
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def agglomerative_cluster_average_linkage(
    dist_matrix: np.ndarray, k: int,
) -> list[list[int]]:
    """Bottom-up agglomerative clustering with average linkage.

    Starts with every point as its own cluster; iteratively merges
    the two closest clusters (by average-linkage distance) until k
    clusters remain.
    """
    n = dist_matrix.shape[0]
    clusters: list[list[int]] = [[i] for i in range(n)]
    D = dist_matrix.astype(np.float64).copy()
    while len(clusters) > k:
        m = len(clusters)
        # Find closest pair (i < j)
        min_d, mi, mj = np.inf, -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                if D[i, j] < min_d:
                    min_d = D[i, j]; mi, mj = i, j
        # Merge
        merged = clusters[mi] + clusters[mj]
        # Remove mj first (larger) then mi
        del clusters[mj]; del clusters[mi]
        clusters.append(merged)
        # Recompute pairwise distances (average linkage from
        # original distance matrix).
        new_m = len(clusters)
        new_D = np.zeros((new_m, new_m))
        for a in range(new_m):
            for b in range(a + 1, new_m):
                ds = [dist_matrix[x, y]
                       for x in clusters[a] for y in clusters[b]]
                new_D[a, b] = new_D[b, a] = float(np.mean(ds))
        D = new_D
    return clusters


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


def _load_model(path):
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
    def __init__(self, base, target): self.base, self.target = base, set(target)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(int(y) in self.target)


def _budget_loader(subset, N, data_dir, seed):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    ds = RelabeledMNIST(base, set(subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=N, replace=False)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=min(64, N),
                       shuffle=False)


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
    model.eval(); pl, ll = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)[:, :n_classes]
            pl.append(F.softmax(logits, dim=1).cpu().numpy())
            ll.append(y.cpu().numpy())
    return np.concatenate(pl), np.concatenate(ll)


def _greedy_coverage(cp, labels, k):
    n = len(cp)
    if k >= n: return list(range(n))
    sel, rem = [], set(range(n))
    running = np.zeros_like(cp[0])
    for _ in range(k):
        best, bi = -1.0, None
        for i in rem:
            trial = (running + cp[i]) / (len(sel) + 1)
            acc = float((trial.argmax(axis=1) == labels).mean())
            if acc > best: best, bi = acc, i
        sel.append(bi); rem.discard(bi)
        running = running + cp[bi]
    return sel


def _ensemble_acc(probs, labels):
    if not probs: return 0.0
    return float((np.mean(probs, axis=0).argmax(axis=1) == labels).mean())


def _spawn(subset, N, seed, data_dir, eval_loader, n_steps=500):
    bl = _budget_loader(subset, N, data_dir, seed)
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(); opt = torch.optim.Adam(m.parameters(), lr=0.01)
    step = 0
    while step < n_steps:
        for x, y in bl:
            if step >= n_steps: break
            opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
            step += 1
    p, l = _probs_and_labels(m, eval_loader, n_classes=2)
    return float((p.argmax(axis=1) == l).mean())


def build_reg(task_dict, traj_dir, probe):
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


def build_sub_registry_from_names(
    task_names: list[str],
    all_entries_by_task: dict[str, list[FractalEntry]],
) -> FractalRegistry:
    """Build a FractalRegistry by taking all seeds of the given task names."""
    reg = FractalRegistry()
    for name in task_names:
        for e in all_entries_by_task.get(name, []):
            reg.add(e)
    return reg


def _anchor_overlap(target: frozenset[int],
                     anchor: frozenset[int]) -> float:
    if not target: return 0.0
    return len(target & anchor) / len(target)


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
    parser.add_argument("--k-coverage", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[50, 100, 300, 1000, 5000])
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=7777)
    parser.add_argument("--budget-seed", type=int, default=2024)
    parser.add_argument("--spawn-seed", type=int, default=2024)
    parser.add_argument("--k-clusters", type=int, nargs="+",
                        default=[2, 3, 4])
    parser.add_argument("--route-threshold", type=float, default=0.75)
    parser.add_argument("--results-out", type=str,
                        default="results/learned_anchors_experiment.json")
    args = parser.parse_args(argv)

    # ── Build two pools to test ──
    # Full: 30 curated (anchor* + anchorB*) + 20 uniform subset_* = 50 tasks
    # Curated-only: 30 curated = 30 tasks (sanity check)
    pool_full: dict[str, tuple[int, ...]] = {}
    pool_full.update(CURATED_A)
    pool_full.update(CURATED_B)
    pool_full.update(UNIFORM_TASKS)

    pool_curated_only: dict[str, tuple[int, ...]] = {}
    pool_curated_only.update(CURATED_A)
    pool_curated_only.update(CURATED_B)

    pools = {"full": pool_full, "curated_only": pool_curated_only}
    print(f"Will test two pools:")
    print(f"  full         : {len(pool_full)} tasks "
          f"({len(CURATED_A)} anchor + {len(CURATED_B)} anchorB + "
          f"{len(UNIFORM_TASKS)} uniform)")
    print(f"  curated_only : {len(pool_curated_only)} tasks "
          f"(no uniform; sanity check)")

    # Loaded once (trajectories are the same regardless of pool)
    pool_tasks_union = {**pool_full}  # union covers everything we need to load

    # ── Load trajectories (for union of all tasks we'll use) ──
    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)
    curated_dir = Path(args.curated_dir)
    sprint7_dir = Path(args.sprint7_dir)

    all_entries_by_task: dict[str, list[FractalEntry]] = {}
    for name in pool_tasks_union:
        traj_dir = (curated_dir if name.startswith("anchor")
                    else sprint7_dir)
        for seed in SEEDS:
            p = traj_dir / f"ext_{name}_seed{seed}_trajectory.npy"
            if not p.is_file(): continue
            sig = _signature(_load_model(p), probe)
            e = FractalEntry(name=f"{name}_seed{seed}", signature=sig,
                              metadata={"task": name, "seed": seed,
                                        "trajectory_path": str(p)})
            all_entries_by_task.setdefault(name, []).append(e)

    # Also load probe entries (for signatures + oracle)
    _, probe_entries = build_reg(ALL_PROBES, curated_dir, probe)
    print(f"Loaded {sum(len(v) for v in all_entries_by_task.values())} "
          f"pool entries, {len(probe_entries)} probe entries.\n")

    # ── Precompute spawn accuracies per (probe, budget) ──
    # Spawn is independent of clustering and pool, so we compute it
    # once per (probe, budget) and reuse across all pool × k
    # combinations. Previously this loop ran ~150 redundant times.
    print("Precomputing spawn accuracies for 15 probes × "
          f"{len(args.budgets)} budgets = {15 * len(args.budgets)} runs...")
    t_spawn = time.time()
    spawn_cache: dict[tuple[str, int], float] = {}
    for probe_name, probe_subset in ALL_PROBES.items():
        eval_loader = _eval_loader(
            probe_subset, args.n_eval, args.data_dir, args.eval_seed)
        for budget in args.budgets:
            acc = _spawn(probe_subset, budget, args.spawn_seed,
                          args.data_dir, eval_loader)
            spawn_cache[(probe_name, budget)] = acc
    print(f"Spawn precompute done in {time.time() - t_spawn:.1f}s "
          f"({len(spawn_cache)} cached values).\n")

    # ── Loop over pools ──
    all_pool_outputs: dict[str, dict] = {}
    for pool_name, pool_tasks in pools.items():
        print("\n" + "#" * 72)
        print(f"  POOL: {pool_name} ({len(pool_tasks)} tasks)")
        print("#" * 72)
        pool_names = sorted(pool_tasks.keys())
        pool_labels = {n: frozenset(pool_tasks[n]) for n in pool_names}

        # ── Pairwise Jaccard distance matrix ──
        n = len(pool_names)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - _jaccard(pool_labels[pool_names[i]],
                                    pool_labels[pool_names[j]])
                D[i, j] = D[j, i] = d

        outputs: dict[int, dict] = {}
        for k in args.k_clusters:
            print("=" * 72)
            print(f"  CLUSTERING k={k} (agglomerative, average-linkage)")
            print("=" * 72)
            clusters_idx = agglomerative_cluster_average_linkage(D, k)
            clusters_idx.sort(key=lambda c: (-len(c), min(c)))
            cluster_members = [
                sorted([pool_names[i] for i in cl]) for cl in clusters_idx
            ]
            cluster_anchors: list[frozenset[int]] = []
            for cl_names in cluster_members:
                counts = {d: 0 for d in range(10)}
                for name in cl_names:
                    for d in pool_labels[name]:
                        counts[d] += 1
                threshold = max(1, len(cl_names) // 2)  # >= 50% majority
                anch = frozenset(
                    d for d, c in counts.items() if c >= threshold)
                cluster_anchors.append(anch)
            for idx, (cl_names, anch) in enumerate(
                    zip(cluster_members, cluster_anchors)):
                hand_label = "_"
                anch_set = set(anch)
                if anch_set <= {0, 1, 2, 3, 4}: hand_label = "⊆A"
                elif anch_set <= {5, 6, 7, 8, 9}: hand_label = "⊆B"
                elif anch_set == set(range(10)): hand_label = "all"
                print(f"  Cluster {idx}  majority-anchor={sorted(anch)}  "
                      f"[{hand_label}]  size={len(cl_names)}")
                print(f"    members: {cl_names[:8]}"
                      f"{'  +more' if len(cl_names) > 8 else ''}")

            print()
            print(f"  PROBE ROUTING (threshold={args.route_threshold})")
            per_probe_rows: list[dict] = []
            for probe_name, probe_subset in ALL_PROBES.items():
                target = frozenset(probe_subset)
                overlaps = [float(_anchor_overlap(target, anch))
                            for anch in cluster_anchors]
                best_i = int(np.argmax(overlaps))
                best_overlap = overlaps[best_i]
                if best_overlap >= args.route_threshold:
                    route = f"cluster_{best_i}"
                else:
                    route = "spawn"

                eval_loader = _eval_loader(
                    probe_subset, args.n_eval, args.data_dir,
                    args.eval_seed)
                ll = []
                for _, y in eval_loader: ll.append(y.numpy())
                eval_labels = np.concatenate(ll)
                q_entry = probe_entries.get((probe_name, 2024))

                per_budget = {}
                if route.startswith("cluster_"):
                    sub_reg = build_sub_registry_from_names(
                        cluster_members[best_i], all_entries_by_task)
                    if len(sub_reg) == 0:
                        per_budget = {b: None for b in args.budgets}
                    else:
                        res = sub_reg.find_nearest(
                            q_entry.signature,
                            k=min(args.candidate_pool, len(sub_reg)))
                        cand_names = [e.name for e in res.entries]
                        cand_models = [_load_model(
                            Path(e.metadata["trajectory_path"]))
                            for e in res.entries]
                        cand_probs_eval = []
                        for cm in cand_models:
                            p, _ = _probs_and_labels(cm, eval_loader,
                                                       n_classes=2)
                            cand_probs_eval.append(p)
                        for budget in args.budgets:
                            bl = _budget_loader(
                                probe_subset, budget,
                                args.data_dir, args.budget_seed)
                            cpb, bl_labels = [], None
                            for cm in cand_models:
                                p, l = _probs_and_labels(
                                    cm, bl, n_classes=2)
                                cpb.append(p)
                                if bl_labels is None: bl_labels = l
                            sel = _greedy_coverage(
                                cpb, bl_labels, k=args.k_coverage)
                            acc = _ensemble_acc(
                                [cand_probs_eval[i] for i in sel],
                                eval_labels)
                            per_budget[budget] = {
                                "acc": acc,
                                "selected": [cand_names[i] for i in sel],
                            }
                else:
                    for budget in args.budgets:
                        # Cached from precompute above — no retraining.
                        acc = spawn_cache[(probe_name, budget)]
                        per_budget[budget] = {"acc": acc, "selected": None}

                per_probe_rows.append({
                    "probe": probe_name,
                    "target": sorted(target),
                    "region_type": PROBE_REGION[probe_name],
                    "overlaps": overlaps,
                    "route": route,
                    "per_budget": {
                        str(b): per_budget[b] for b in args.budgets
                    },
                })

                print(f"    {probe_name:<17s} region="
                      f"{PROBE_REGION[probe_name]:<5s} "
                      f"target={sorted(target)}  overlaps="
                      f"[{','.join(f'{o:.2f}' for o in overlaps)}]  "
                      f"→ {route}")

            print()
            print(f"  AGGREGATE BY REGION TYPE at k={k}:")
            for rt in ["in_A", "in_B", "cross"]:
                rows = [r for r in per_probe_rows if r["region_type"] == rt]
                if not rows: continue
                row_line = f"    {rt:<6s}  "
                for budget in args.budgets:
                    accs = [r["per_budget"][str(budget)]["acc"]
                            for r in rows
                            if r["per_budget"][str(budget)] is not None]
                    if accs:
                        row_line += f"N={budget}:{np.mean(accs):.3f}  "
                print(row_line)

            outputs[k] = {
                "cluster_anchors": [sorted(a) for a in cluster_anchors],
                "cluster_members": cluster_members,
                "per_probe_rows": per_probe_rows,
            }
            print()

        # Per-pool verdict
        print("=" * 72)
        print(f"  VERDICT for pool '{pool_name}'")
        print("=" * 72)
        if 2 in outputs:
            anchors_2 = [set(a) for a in outputs[2]["cluster_anchors"]]
            hand_A = set(range(5)); hand_B = set(range(5, 10))
            match_A = any(a <= hand_A and a for a in anchors_2)
            match_B = any(a <= hand_B and a for a in anchors_2)
            has_wide = any(a == set(range(10)) for a in anchors_2)
            print(f"  k=2 majority-anchors: "
                  f"{[sorted(a) for a in anchors_2]}")
            if match_A and match_B and not has_wide:
                print("  → Learned anchors MATCH hand-picked {0-4}/{5-9}.")
            elif has_wide:
                print("  → Learned anchors include a 'universal' cluster.")
            else:
                print("  → Learned anchors DIFFER from hand-picked.")
            probe_rows = outputs[2]["per_probe_rows"]
            for rt in ["in_A", "in_B", "cross"]:
                rts = [r["route"] for r in probe_rows
                       if r["region_type"] == rt]
                print(f"  {rt} routes: {rts}")

        all_pool_outputs[pool_name] = outputs

    out = {
        "k_clusters": args.k_clusters,
        "route_threshold": args.route_threshold,
        "pools": {
            pool_name: {
                str(k): {
                    "cluster_anchors":
                        [sorted(a) for a in v["cluster_anchors"]],
                    "cluster_members": v["cluster_members"],
                    "per_probe_rows": v["per_probe_rows"],
                }
                for k, v in po.items()
            }
            for pool_name, po in all_pool_outputs.items()
        },
    }
    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults saved: {args.results_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
