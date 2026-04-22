"""v3 Sprint 18 — fractal audit driver.

Measures correlation dimension + scale stability on three structural
objects of the FractalTrainer registry:

  1. Signature point cloud  (the N signatures as points in R^1000)
  2. Growth trajectory      (N_query signatures produced by oracles
                             on a stream of distinct-task queries)
  3. Label-set lattice      (binary-indicator vectors of each
                             registered expert's class-1 set in R^10)

For each object: correlation dim, R², scale-stability (slope across
small/mid/large r-bands), and a random-baseline dim at matched (N, D).
Verdict classifies each object as pass / weak_pass / fail per the
acceptance criteria in Review 42's plan.

Run:
    python scripts/run_fractal_audit.py --mode smoke   # ~2 min
    python scripts/run_fractal_audit.py --mode full    # ~15-30 min
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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fractaltrainer.integration import (  # noqa: E402
    audit_label_lattice, audit_signature_cloud, audit_trajectory,
    classify_verdict,
)
from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402

from run_fractal_demo import (  # noqa: E402
    _probe, _train_loader, _probe_signature,
    _make_untrained_model, _train_seed_expert,
)


def _enumerate_binary_tasks(
    n_classes: int = 10, target_n: int = 60, seed: int = 7,
) -> list[tuple[str, tuple[int, ...]]]:
    """Enumerate distinct class-1 label subsets up to complement.

    Takes subsets of sizes {3..7}, drops any subset whose complement
    is also in the list (symmetry), shuffles, returns the first
    target_n.
    """
    cands: list[tuple[int, ...]] = []
    seen_canonical: set[frozenset] = set()
    for k in (3, 4, 5, 6, 7):
        for c in itertools.combinations(range(n_classes), k):
            fc = frozenset(c)
            fc_comp = frozenset(range(n_classes)) - fc
            canonical = min(fc, fc_comp, key=lambda s: tuple(sorted(s)))
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            cands.append(tuple(sorted(c)))
    rng = random.Random(seed)
    rng.shuffle(cands)
    picked = cands[:target_n]
    return [(f"subset_{''.join(str(d) for d in s)}", s) for s in picked]


def _build_audit_registry(
    tasks: list[tuple[str, tuple[int, ...]]],
    probe: torch.Tensor, data_dir: str,
    seed_steps: int, train_size: int, batch_size: int,
    dataset: str, arch: str,
    seeds: list[int],
    verbose: bool = True,
) -> tuple[FractalRegistry, dict[str, torch.nn.Module]]:
    """Train one expert per (task, seed) pair; build a flat registry."""
    reg = FractalRegistry()
    models: dict[str, torch.nn.Module] = {}
    total = len(tasks) * len(seeds)
    i = 0
    for task_name, target in tasks:
        for s in seeds:
            i += 1
            if verbose:
                print(f"  [{i}/{total}] {task_name} seed={s}")
            m = _train_seed_expert(
                target, s, seed_steps, train_size, batch_size,
                data_dir, dataset=dataset, arch=arch,
            )
            sig = _probe_signature(m, probe)
            name = f"{task_name}_seed{s}"
            reg.add(FractalEntry(
                name=name, signature=sig,
                metadata={
                    "task": task_name,
                    "task_labels": list(sorted(target)),
                    "seed": s,
                },
            ))
            models[name] = m
    return reg, models


def _generate_trajectory(
    n: int, arch: str, probe: torch.Tensor, data_dir: str,
    dataset: str, train_size: int, batch_size: int,
    n_oracle_steps: int = 100, seed: int = 42,
    verbose: bool = True,
) -> list[np.ndarray]:
    """Stream n distinct query tasks; for each, train a short oracle
    and signature it. Returns the sequence of signatures."""
    rng = random.Random(seed)
    all_tasks = _enumerate_binary_tasks(target_n=max(n * 2, 80), seed=seed + 1)
    rng.shuffle(all_tasks)
    trajectory: list[np.ndarray] = []
    for i, (task_name, target) in enumerate(all_tasks[:n]):
        if verbose and i % 10 == 0:
            print(f"  trajectory step {i}/{n} — {task_name}")
        train_ldr = _train_loader(
            target, train_size, batch_size, data_dir,
            seed=9000 + i, dataset=dataset,
        )
        torch.manual_seed(13 + i)
        oracle = _make_untrained_model(arch)
        opt = torch.optim.Adam(oracle.parameters(), lr=0.01)
        it = iter(train_ldr)
        for _ in range(n_oracle_steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_ldr); x, y = next(it)
            logits = oracle(x, context=None)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        trajectory.append(_probe_signature(oracle, probe))
    return trajectory


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--data-dir", default="results/data")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--arch", default="mlp")
    parser.add_argument("--n-registry-tasks", type=int, default=None,
                        help="distinct tasks in registry (full mode default 20)")
    parser.add_argument("--n-trajectory", type=int, default=None,
                        help="query-stream length (full mode default 50)")
    parser.add_argument("--registry-seeds", type=int, nargs="+",
                        default=[42, 101, 2024])
    parser.add_argument("--seed-steps", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--oracle-steps", type=int, default=100)
    parser.add_argument("--results-out",
                        default="results/fractal_audit.json")
    parser.add_argument("--plot-out",
                        default="results/fractal_audit_scaling.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    is_smoke = args.mode == "smoke"
    if args.n_registry_tasks is None:
        args.n_registry_tasks = 3 if is_smoke else 20
    if args.n_trajectory is None:
        args.n_trajectory = 5 if is_smoke else 50
    if args.seed_steps is None:
        args.seed_steps = 40 if is_smoke else 500
    if is_smoke:
        args.registry_seeds = args.registry_seeds[:1]
        args.train_size = min(args.train_size, 512)
        args.oracle_steps = min(args.oracle_steps, 40)
    registry_seeds: list[int] = args.registry_seeds

    print("=" * 72)
    print(f"  FRACTAL AUDIT — {args.mode} mode, {args.dataset} {args.arch}")
    print(f"  registry: {args.n_registry_tasks} tasks × "
          f"{len(registry_seeds)} seeds; trajectory: {args.n_trajectory}")
    print("=" * 72)

    t_start = time.time()

    # ── Build the registry ──
    print(f"\n[1/4] building registry "
          f"(n_steps={args.seed_steps}, train_size={args.train_size})...")
    probe = _probe(args.data_dir, args.n_probe, args.probe_seed,
                    dataset=args.dataset)
    tasks = _enumerate_binary_tasks(target_n=args.n_registry_tasks)
    registry, models = _build_audit_registry(
        tasks, probe, args.data_dir, args.seed_steps,
        args.train_size, args.batch_size,
        dataset=args.dataset, arch=args.arch,
        seeds=registry_seeds,
    )
    print(f"  registry: {len(registry)} entries")

    # ── Generate trajectory ──
    print(f"\n[2/4] generating growth trajectory "
          f"({args.n_trajectory} queries)...")
    trajectory = _generate_trajectory(
        args.n_trajectory, args.arch, probe, args.data_dir, args.dataset,
        args.train_size, args.batch_size,
        n_oracle_steps=args.oracle_steps, seed=args.probe_seed + 1,
    )

    # ── Audit each object ──
    print("\n[3/4] auditing each object...")
    sig_audit = audit_signature_cloud(registry, random_seed=0)
    traj_audit = audit_trajectory(trajectory, random_seed=0)
    label_sets = [
        frozenset(e.metadata.get("task_labels") or [])
        for e in registry.entries()
    ]
    lattice_audit = audit_label_lattice(
        label_sets, n_classes=10, random_seed=0,
    )

    audits = [sig_audit, traj_audit, lattice_audit]
    for a in audits:
        cd = a.correlation_dim
        st = a.scale_stability
        print(f"  {a.name:<22s}  N={a.n_points:>3d}  "
              f"D={cd.dim:+.3f}  R²={cd.r_squared:.3f}  "
              f"slope_var={st.slope_variance:.3f}  "
              f"baseline_D={a.random_baseline_dim.dim:+.3f}")

    # ── Verdict ──
    print("\n[4/4] verdict")
    verdict = classify_verdict(audits)
    for per_obj in verdict["per_object"]:
        print(f"  {per_obj['name']:<22s}  D={per_obj['dim']:+.3f}  "
              f"sv={per_obj['slope_variance']}  "
              f"→ {per_obj['classification'].upper()}")
    print(f"\n  OVERALL: {verdict['overall'].upper()}")

    recommendation = _recommend_phase2(verdict, audits)
    print(f"  PHASE 2 RECOMMENDATION: {recommendation}")

    # ── Plot ──
    if not args.no_plot:
        _plot_scaling(audits, Path(args.plot_out))

    # ── Save ──
    elapsed = time.time() - t_start
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "config": vars(args),
            "elapsed_s": elapsed,
            "registry_size": len(registry),
            "audits": [a.to_dict() for a in audits],
            "verdict": verdict,
            "phase2_recommendation": recommendation,
        }, f, indent=2, default=str)
    print(f"\n  elapsed: {elapsed:.1f}s")
    print(f"  saved: {out_path}")
    return 0


def _recommend_phase2(verdict: dict, audits: list) -> str:
    """Pick the Phase 2 action per the plan table."""
    overall = verdict["overall"]
    if overall == "fail":
        return "rename (architecture does not exhibit fractal structure)"
    if overall == "weak_pass":
        return "rename (variance 0.3-0.6 implies weak fractality; paper shouldn't hedge)"
    # overall == pass — pick based on WHICH object passed
    sig_cls = next(
        (p for p in verdict["per_object"] if p["name"] == "signature_cloud"),
        None,
    )
    lattice_cls = next(
        (p for p in verdict["per_object"] if p["name"] == "label_lattice"),
        None,
    )
    if sig_cls and sig_cls["classification"] == "pass":
        return ("ifs_spawn — signature cloud has fractal structure; "
                "contractive-affine spawn could generate in that attractor")
    if lattice_cls and lattice_cls["classification"] == "pass":
        return ("recursive_registry — label lattice has fractal structure; "
                "a depth-≥3 hierarchical registry could match it")
    return ("ifs_spawn_or_recursive — fractal structure found on "
            "trajectory only; direction ambiguous, default to IFS")


def _plot_scaling(audits: list, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable — skipping plot")
        return
    fig, axes = plt.subplots(1, len(audits), figsize=(5 * len(audits), 4))
    if len(audits) == 1:
        axes = [axes]
    for ax, a in zip(axes, audits):
        cd = a.correlation_dim
        if cd.radii.size == 0:
            ax.set_title(f"{a.name}\n(no data)")
            continue
        mask = cd.correlation_sums > 0
        lr = np.log(cd.radii[mask])
        lc = np.log(cd.correlation_sums[mask])
        ax.plot(lr, lc, "o-", label="data", markersize=4)
        # Highlight scaling window
        s, e = cd.scaling_start, cd.scaling_end
        if 0 <= s < e <= len(lr):
            xw, yw = lr[s:e], lc[s:e]
            slope, intercept = np.polyfit(xw, yw, 1)
            ax.plot(xw, slope * xw + intercept, "r--",
                     label=f"D={slope:.3f}", linewidth=2)
        # Random baseline
        rb = a.random_baseline_dim
        if rb.radii.size > 0:
            rm = rb.correlation_sums > 0
            ax.plot(np.log(rb.radii[rm]), np.log(rb.correlation_sums[rm]),
                     "gray", alpha=0.4,
                     label=f"random baseline D={rb.dim:.2f}")
        ax.set_xlabel("log r")
        ax.set_ylabel("log C(r)")
        ax.set_title(f"{a.name} (N={a.n_points})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  plot saved: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
