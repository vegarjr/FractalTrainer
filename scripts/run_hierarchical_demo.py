"""v3 Sprint 17 — Direction E demo: flat vs hierarchical routing.

HierarchicalRegistry (Sprint 8) was originally designed for
dataset-coarse routing: one sub-registry per dataset, then flat
nearest-neighbor search within. It turns out the same infrastructure
works as a generic two-level tree keyed by any metadata field —
including `task`.

For a 15-expert flat registry (5 tasks × 3 seeds), a hierarchical
view with `dataset_key="task"` partitions into 5 task-branches of 3
seed-leaves each. At routing time:

    Flat:          15 L2 distance computations
    Hierarchical:  5 centroid distances + 3 in-branch distances = 8

The demo measures:
  1. Decision consistency (does hierarchical pick the same entry as
     flat, for each query?)
  2. Latency delta (flat dt vs hier dt, averaged over many queries)
  3. What hierarchical does DIFFERENTLY — when flat finds a nearest
     in one task but the signature actually belongs to a neighbor
     task, hierarchical's coarse gate can re-route.

Run:
    python scripts/run_hierarchical_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fractaltrainer.registry import (  # noqa: E402
    FractalEntry, FractalRegistry, HierarchicalRegistry,
)

from run_fractal_demo import (  # noqa: E402
    _build_seed_registry, _probe, _train_loader, _probe_signature,
    _make_untrained_model,
)


QUERY_SPECS = [
    # (name, task_for_oracle, class-1 label set, oracle_seed)
    ("Q_match_odds",        (1, 3, 5, 7, 9), 9101),
    ("Q_match_lows",        (0, 1, 2, 3, 4), 9102),
    ("Q_match_evens_low",   (0, 2, 4),       9103),
    ("Q_match_highs",       (5, 6, 7, 8, 9), 9104),
    ("Q_match_midodds",     (3, 5, 7),       9105),
    # Cross-task queries that expose coarse-routing differences
    ("Q_bridge_lowhigh_05", (0, 5),          9201),
    ("Q_bridge_mixodd_357", (3, 5, 7),       9202),  # same as midodds
    ("Q_bridge_evens_048",  (0, 4, 8),       9203),
]


def _oracle_signature(target, train_loader, probe, arch, n_steps=100):
    torch.manual_seed(13)
    oracle = _make_untrained_model(arch)
    opt = torch.optim.Adam(oracle.parameters(), lr=0.01)
    it = iter(train_loader)
    for _ in range(n_steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        logits = oracle(x, context=None)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return _probe_signature(oracle, probe)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="results/data")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--arch", default="mlp")
    parser.add_argument("--seed-steps", type=int, default=500)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--latency-trials", type=int, default=100)
    parser.add_argument("--results-out",
                        default="results/hierarchical_demo.json")
    args = parser.parse_args(argv)

    print("=" * 72)
    print(f"  HIERARCHICAL DEMO — flat vs 2-level tree on {args.dataset}")
    print("=" * 72)

    print("\n[1/4] loading probe + seed registry...")
    probe = _probe(args.data_dir, args.n_probe, args.probe_seed,
                    dataset=args.dataset)
    flat_reg, models = _build_seed_registry(
        probe, args.data_dir, args.seed_steps,
        args.train_size, args.batch_size, verbose=True,
        dataset=args.dataset, arch=args.arch,
    )
    print(f"  flat registry: {len(flat_reg)} entries")

    # Build a hierarchical copy: keyed by `task` metadata
    hier_reg = HierarchicalRegistry(dataset_key="task")
    for entry in flat_reg.entries():
        hier_reg.add(entry)
    print(f"  hierarchical: {len(hier_reg)} entries across "
          f"{len(hier_reg.datasets())} task-branches: "
          f"{hier_reg.datasets()}")

    # ── Compute query signatures ──
    print("\n[2/4] computing signatures for 8 test queries...")
    query_sigs: dict[str, np.ndarray] = {}
    for name, target, oracle_seed in QUERY_SPECS:
        loader = _train_loader(target, args.train_size, args.batch_size,
                                args.data_dir, oracle_seed,
                                dataset=args.dataset)
        sig = _oracle_signature(target, loader, probe, args.arch)
        query_sigs[name] = sig

    # ── Compare decisions ──
    print("\n[3/4] comparing flat vs hierarchical decisions...")
    comparisons: list[dict] = []
    agreements = 0
    for name, sig in query_sigs.items():
        flat_res = flat_reg.find_nearest(sig, k=1, query_name=name)
        hier_res = hier_reg.find_nearest_hierarchical(sig, k=1, query_name=name)

        flat_pick = flat_res.nearest.name if flat_res.nearest else None
        hier_pick = hier_res.retrieval.nearest.name if hier_res.retrieval.nearest else None
        flat_d = flat_res.distances[0] if flat_res.distances else None
        hier_d = hier_res.retrieval.distances[0] if hier_res.retrieval.distances else None

        agree = flat_pick == hier_pick
        agreements += int(agree)

        comparisons.append({
            "query": name,
            "flat_pick": flat_pick, "flat_distance": flat_d,
            "hier_pick": hier_pick, "hier_distance": hier_d,
            "hier_chosen_branch": hier_res.chosen_dataset,
            "hier_centroid_distance": hier_res.centroid_distance,
            "all_centroid_distances": hier_res.all_centroid_distances,
            "agree": agree,
        })
        print(f"  {name:<24s}  flat→{flat_pick}  "
              f"hier→{hier_pick} (branch={hier_res.chosen_dataset})  "
              f"{'✓' if agree else '✗'}")

    print(f"\n  decisions agree: {agreements}/{len(query_sigs)}")

    # ── Latency comparison ──
    print(f"\n[4/4] measuring latency ({args.latency_trials} trials each)...")
    # Use one query signature for latency microbench
    sig = next(iter(query_sigs.values()))

    t0 = time.perf_counter()
    for _ in range(args.latency_trials):
        _ = flat_reg.find_nearest(sig, k=1)
    flat_dt = (time.perf_counter() - t0) / args.latency_trials

    t0 = time.perf_counter()
    for _ in range(args.latency_trials):
        _ = hier_reg.find_nearest_hierarchical(sig, k=1)
    hier_dt = (time.perf_counter() - t0) / args.latency_trials

    # Theoretical distance-op count
    n_flat_ops = len(flat_reg)
    n_hier_ops = len(hier_reg.datasets()) + max(
        len(hier_reg.sub_registry(ds) or FractalRegistry())
        for ds in hier_reg.datasets()
    )
    print(f"\n  flat mean latency:  {flat_dt*1000:.3f} ms  ({n_flat_ops} distance ops)")
    print(f"  hier mean latency:  {hier_dt*1000:.3f} ms  ({n_hier_ops} distance ops)")
    speedup = flat_dt / hier_dt if hier_dt > 0 else float("inf")
    print(f"  speedup:            {speedup:.2f}×")

    # Save
    out = {
        "config": vars(args),
        "n_entries": len(flat_reg),
        "n_branches": len(hier_reg.datasets()),
        "branches": hier_reg.datasets(),
        "comparisons": comparisons,
        "n_agreements": agreements,
        "n_queries": len(query_sigs),
        "latency": {
            "flat_ms_per_query": flat_dt * 1000,
            "hier_ms_per_query": hier_dt * 1000,
            "speedup": speedup,
            "n_flat_distance_ops": n_flat_ops,
            "n_hier_distance_ops": n_hier_ops,
            "trials": args.latency_trials,
        },
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
