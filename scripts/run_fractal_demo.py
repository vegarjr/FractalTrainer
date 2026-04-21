"""v3 Sprint 17 — Fractal showcase: closed perception→growth loop with context injection.

End-to-end demo:
    1. Boot 5 seed tasks × 3 seeds = 15 ContextAwareMLP experts on
       MNIST binary subsets {0-4, 5-9, evens-low, odds, mid-odds}.
    2. Present three unseen-task queries:
          Q_match   — same task as a seed, different random seed
          Q_compose — evens (={0,2,4,6,8}), no single expert covers it
          Q_spawn   — {0,1,9}, Jaccard ≤ 0.25 vs every seed expert
    3. On the spawn query, run three-arm ablation:
          A. no context          (baseline)
          B. K=3 nearest context (the C contribution)
          C. K=3 random context  (control — isolates routing from aux input)
       Across budgets N ∈ {50, 100, 300, 500, 1000} and 3 seeds.
    4. Emit markdown tables + results/fractal_demo.json + optional PNG.

Run:
    python scripts/run_fractal_demo.py --mode smoke --llm mock
    python scripts/run_fractal_demo.py --mode full --llm mock --plot

The describer is mocked by default so routing behavior is isolated
from LLM perception noise. Pass `--llm local` to use Qwen, `--llm cli`
for Claude.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.integration import (  # noqa: E402
    ContextAwareMLP, ContextSpec,
    Describer, FractalPipeline, MockDescriber, OracleDescriber,
    QueryInput, ReclusterPolicy, SampleEfficiencyResult,
    evaluate_expert, render_efficiency_table_md, sample_efficiency_curve,
    spawn_baseline, spawn_random_context, spawn_with_context,
)
from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402


SEED_TASKS: dict[str, tuple[int, ...]] = {
    "subset_01234": (0, 1, 2, 3, 4),
    "subset_56789": (5, 6, 7, 8, 9),
    "subset_024":   (0, 2, 4),
    "subset_13579": (1, 3, 5, 7, 9),
    "subset_357":   (3, 5, 7),
}
SEEDS = [42, 101, 2024]

QUERY_TASKS = {
    "Q_match":   ("subset_13579", (1, 3, 5, 7, 9), 9001),
    "Q_compose": ("subset_02468", (0, 2, 4, 6, 8), 9002),
    "Q_spawn":   ("subset_019",   (0, 1, 9),        9003),
}


class RelabeledMNIST(Dataset):
    def __init__(self, base, target):
        self.base = base
        self.target = set(int(d) for d in target)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(int(y) in self.target)


def _mnist_probe(data_dir: str, n: int, seed: int) -> torch.Tensor:
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(base), size=n, replace=False)
    return torch.stack([base[i][0] for i in idx.tolist()], dim=0)


def _train_loader(target: tuple[int, ...], n: int, batch_size: int,
                   data_dir: str, seed: int) -> DataLoader:
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    ds = RelabeledMNIST(base, target)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    return DataLoader(Subset(ds, idx.tolist()),
                       batch_size=batch_size, shuffle=True, drop_last=True)


def _eval_loader(target: tuple[int, ...], n: int, data_dir: str,
                  seed: int) -> DataLoader:
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    ds = RelabeledMNIST(base, target)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=64, shuffle=False)


def _sample_pairs(target: tuple[int, ...], n_pairs: int,
                   data_dir: str, seed: int) -> list[tuple[int, int]]:
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(base), size=n_pairs, replace=False)
    ts = set(int(x) for x in target)
    out = []
    for i in idx.tolist():
        _, d = base[i]
        d = int(d)
        out.append((d, int(d in ts)))
    return out


def _probe_signature(model: torch.nn.Module, probe: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        try:
            logits = model(probe, context=None)
        except TypeError:
            logits = model(probe)
        p = F.softmax(logits, dim=1)
    return p.flatten().cpu().numpy()


def _train_seed_expert(
    target: tuple[int, ...], seed: int, n_steps: int,
    train_size: int, batch_size: int, data_dir: str,
) -> ContextAwareMLP:
    """Train a seed expert with context_scale=0 (baseline MLP behavior)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ContextAwareMLP(context_scale=0.0)
    loader = _train_loader(target, train_size, batch_size, data_dir, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    step = 0
    it = iter(loader)
    while step < n_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        logits = model(x, context=None)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        step += 1
    return model


def _build_seed_registry(
    probe: torch.Tensor, data_dir: str, seed_steps: int,
    train_size: int, batch_size: int, verbose: bool,
) -> tuple[FractalRegistry, dict[str, torch.nn.Module]]:
    registry = FractalRegistry()
    models: dict[str, torch.nn.Module] = {}
    total = len(SEED_TASKS) * len(SEEDS)
    i = 0
    t0 = time.time()
    for task_name, target in SEED_TASKS.items():
        for seed in SEEDS:
            i += 1
            if verbose:
                print(f"  [{i}/{total}] training seed expert: {task_name} seed={seed}")
            model = _train_seed_expert(
                target, seed, seed_steps, train_size, batch_size, data_dir,
            )
            sig = _probe_signature(model, probe)
            name = f"{task_name}_seed{seed}"
            entry = FractalEntry(
                name=name, signature=sig,
                metadata={
                    "task": task_name,
                    "task_labels": list(sorted(target)),
                    "seed": seed,
                    "spawned": False,
                    "context_mode": "none",
                },
            )
            registry.add(entry)
            models[name] = model
    elapsed = time.time() - t0
    if verbose:
        print(f"  total registry bootstrapping: {elapsed:.1f}s for {len(registry)} entries")
    return registry, models


def _ablation_for_spawn_query(
    q: QueryInput, registry: FractalRegistry,
    model_by_entry: dict[str, torch.nn.Module],
    *, budgets: list[int], seeds: list[int],
    spec: ContextSpec, context_scale: float,
    lr: float,
    verbose: bool,
) -> dict[str, SampleEfficiencyResult]:
    """Run three-arm (A=none, B=neighbors, C=random) ablation across budgets × seeds."""
    # Find K nearest signatures to the spawn query's truth
    from fractaltrainer.integration import evaluate_expert as _eval
    # We need the decision's retrieval to know neighbor models + distances.
    # Use a synthetic signature: train a tiny "oracle" model (context=None)
    # on the spawn task and signature it once. The ablation uses the SAME
    # nearest set for every run — critical for comparability.
    probe = q.probe
    # Compute "oracle" signature by training one expert with a modest budget
    # once, just to get a representative signature for routing.
    torch.manual_seed(13)
    oracle = ContextAwareMLP(context_scale=0.0)
    opt = torch.optim.Adam(oracle.parameters(), lr=0.01)
    it = iter(q.train_loader)
    for _ in range(100):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(q.train_loader); x, y = next(it)
        logits = oracle(x, context=None)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    q_sig = _probe_signature(oracle, probe)

    decision = registry.decide(
        q_sig, match_threshold=5.0, spawn_threshold=7.0, compose_k=spec.k,
    )
    nearest_entries = list(decision.retrieval.entries[:spec.k]) if decision.retrieval else []
    nearest_models = [model_by_entry[e.name] for e in nearest_entries]
    nearest_distances = list(decision.retrieval.distances[:spec.k]) if decision.retrieval else []

    if verbose:
        print(f"    [ablation] routing verdict: {decision.verdict} "
              f"min_distance={decision.min_distance:.3f}")
        print(f"    [ablation] K={spec.k} neighbors: "
              f"{[e.name for e in nearest_entries]}")

    eval_batches = list(q.eval_loader)

    def make_arm_A(n_steps, seed):
        _, _, stats = spawn_baseline(
            q.train_loader, probe, n_steps=n_steps, lr=lr, seed=seed,
            entry_name=f"spawnA_{q.name}_seed{seed}",
            task=q.name,
        )
        # Return trained model directly via side-channel — wrap:
        model = _last_trained_model["A"]
        return model, stats.final_loss

    # Need to capture the model. Simpler: inline the arm without needing
    # the helper — but the signature required by sample_efficiency_curve is
    # (budget, seed) -> (model, loss). Do the wrapping manually:
    def run_arm_baseline(budget, seed):
        m, _, stats = spawn_baseline(
            q.train_loader, probe, n_steps=budget, lr=lr, seed=seed,
            entry_name=f"spawnA_{q.name}_seed{seed}_N{budget}",
            task=q.name,
        )
        return m, stats.final_loss

    def run_arm_neighbors(budget, seed):
        m, _, stats = spawn_with_context(
            q.train_loader, probe,
            neighbors=nearest_models, neighbor_distances=nearest_distances,
            spec=spec, context_scale=context_scale,
            n_steps=budget, lr=lr, seed=seed,
            entry_name=f"spawnB_{q.name}_seed{seed}_N{budget}",
            task=q.name,
        )
        return m, stats.final_loss

    def run_arm_random(budget, seed):
        m, _, stats = spawn_random_context(
            q.train_loader, probe,
            context_scale=context_scale,
            n_steps=budget, lr=lr, seed=seed,
            entry_name=f"spawnC_{q.name}_seed{seed}_N{budget}",
            task=q.name,
        )
        return m, stats.final_loss

    results: dict[str, SampleEfficiencyResult] = {}
    for arm_name, runner in [
        ("A_no_context", run_arm_baseline),
        ("B_nearest_context", run_arm_neighbors),
        ("C_random_context", run_arm_random),
    ]:
        if verbose:
            print(f"    [ablation] arm {arm_name} × budgets {budgets} × seeds {seeds}")
        r = sample_efficiency_curve(
            runner, arm_name=arm_name,
            budgets=budgets, seeds=seeds, eval_loader=eval_batches,
        )
        results[arm_name] = r
    return results


_last_trained_model: dict[str, torch.nn.Module] = {}


def _compose_accuracy(
    decision, model_by_entry: dict[str, torch.nn.Module],
    eval_batches: list,
) -> float:
    """Accuracy of coverage-weighted ensemble over decision.composite_entries."""
    if not decision.composite_entries:
        return float("nan")
    probs_list = []
    labels_list = []
    for entry, w in zip(decision.composite_entries, decision.composite_weights or [1.0] * len(decision.composite_entries)):
        model = model_by_entry.get(entry.name)
        if model is None:
            continue
        model.eval()
        batch_probs = []
        batch_labels = []
        with torch.no_grad():
            for x, y in eval_batches:
                try:
                    logits = model(x, context=None)
                except TypeError:
                    logits = model(x)
                # Use first two class logits (binary task)
                p = F.softmax(logits[:, :2], dim=1).cpu().numpy()
                batch_probs.append(p)
                batch_labels.append(y.cpu().numpy())
        if not batch_probs:
            continue
        probs_list.append((float(w), np.concatenate(batch_probs)))
        labels_list = np.concatenate(batch_labels)
    if not probs_list:
        return float("nan")
    total_w = sum(w for w, _ in probs_list)
    blend = sum((w / total_w) * p for w, p in probs_list)
    return float((blend.argmax(axis=1) == labels_list).mean())


def _match_accuracy(
    entry_name: str, model_by_entry: dict[str, torch.nn.Module],
    eval_batches: list,
) -> float:
    model = model_by_entry[entry_name]
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in eval_batches:
            try:
                logits = model(x, context=None)
            except TypeError:
                logits = model(x)
            pred = logits[:, :2].argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def _build_describer(llm: str, local_url: str):
    if llm == "mock":
        return OracleDescriber()
    if llm == "local":
        from fractaltrainer.repair.llm_client import make_local_llm_client
        fn = make_local_llm_client(base_url=local_url, temperature=0.3, max_tokens=256)
        return Describer(fn)
    if llm == "cli":
        from fractaltrainer.repair.llm_client import make_claude_cli_client
        return Describer(make_claude_cli_client())
    raise ValueError(f"unknown --llm: {llm!r}")


def _plot_efficiency(results: dict[str, SampleEfficiencyResult],
                     budgets: list[int], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm_name, r in results.items():
        means = [r.mean_by_budget[b] for b in budgets]
        stdevs = [r.stdev_by_budget[b] for b in budgets]
        ax.errorbar(budgets, means, yerr=stdevs, marker="o", label=arm_name,
                    capsize=3)
    ax.set_xlabel("Training steps (N)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Spawn-path sample efficiency: context ablation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  plot saved: {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--llm", choices=("mock", "local", "cli"), default="mock")
    parser.add_argument("--local-llm-url", default="http://127.0.0.1:8080")
    parser.add_argument("--data-dir", default="results/data")
    parser.add_argument("--results-out", default="results/fractal_demo.json")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-out", default="results/fractal_demo.png")
    parser.add_argument("--context-scale", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=3,
                        help="number of nearest neighbors for context")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024])
    parser.add_argument("--seed-steps", type=int, default=None,
                        help="training steps for the 15 seed experts")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--match-threshold", type=float, default=5.0)
    parser.add_argument("--spawn-threshold", type=float, default=7.0)
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--budgets", type=int, nargs="+", default=None,
                        help="ablation budgets (defaults by mode)")
    parser.add_argument("--n-pairs", type=int, default=20,
                        help="(digit, label) pairs shown to describer")
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    is_smoke = args.mode == "smoke"
    seed_steps = args.seed_steps or (50 if is_smoke else 500)
    if args.budgets is None:
        args.budgets = [50, 100] if is_smoke else [50, 100, 300, 500, 1000]
    if is_smoke:
        args.seeds = args.seeds[:1]

    verbose = args.verbose or (not is_smoke)
    print("=" * 72)
    print(f"  FRACTAL DEMO — mode={args.mode}  llm={args.llm}  K={args.K}  "
          f"scale={args.context_scale}")
    print("=" * 72)

    # ── Build probe ──
    print("\n[1/4] loading MNIST probe batch...")
    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)

    # ── Build seed registry ──
    print(f"\n[2/4] training {len(SEED_TASKS) * len(SEEDS)} seed experts "
          f"(n_steps={seed_steps}, train_size={args.train_size})...")
    registry, models = _build_seed_registry(
        probe, args.data_dir, seed_steps,
        args.train_size, args.batch_size, verbose,
    )

    # ── Pipeline ──
    describer = _build_describer(args.llm, args.local_llm_url)
    context_spec = ContextSpec(k=args.K, aggregation="weighted_mean")
    pipeline = FractalPipeline(
        registry, describer,
        match_threshold=args.match_threshold,
        spawn_threshold=args.spawn_threshold,
        compose_k=args.K,
        context_spec=context_spec,
        context_scale=args.context_scale,
        model_by_entry=models,
        recluster_policy=ReclusterPolicy(interval_spawns=2, trigger_at_end=True),
    )

    # ── Three queries ──
    print("\n[3/4] running 3 queries: match, compose, spawn...")
    steps_out: list[dict] = []
    main_table_rows: list[tuple[str, str, str, float, float]] = []
    ablation_by_query: dict[str, dict] = {}

    for q_name, (task_name, labels, seed) in QUERY_TASKS.items():
        print(f"\n  ── {q_name} — task={task_name} labels={labels} ──")
        pairs = _sample_pairs(labels, args.n_pairs, args.data_dir, seed)
        train_ldr = _train_loader(labels, args.train_size, args.batch_size,
                                    args.data_dir, seed)
        eval_ldr = list(_eval_loader(labels, args.n_eval, args.data_dir,
                                       seed + 1))
        q = QueryInput(
            name=q_name, pairs=pairs, truth_labels=frozenset(labels),
            train_loader=train_ldr, eval_loader=eval_ldr, probe=probe,
        )

        # Signature the query: train a small oracle model, signature it.
        torch.manual_seed(seed)
        oracle = ContextAwareMLP(context_scale=0.0)
        opt = torch.optim.Adam(oracle.parameters(), lr=args.lr)
        it = iter(train_ldr)
        for _ in range(100 if not is_smoke else 40):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_ldr); x, y = next(it)
            logits = oracle(x, context=None)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        q_sig = _probe_signature(oracle, probe)

        # Execute one pipeline step for verdict + accuracy logging
        step = pipeline.step(
            q, signature=q_sig, spawn_mode="neighbors",
            spawn_n_steps=max(args.budgets), spawn_lr=args.lr,
            spawn_seed=args.seeds[0],
        )

        # Measure accuracy per verdict
        accuracy = step.accuracy
        efficiency = None
        if step.verdict == "match":
            accuracy = _match_accuracy(step.neighbors_used[0] if step.neighbors_used
                                        else step.decision.retrieval.entries[0].name,
                                        models, eval_ldr)
            action_str = f"route → {step.neighbors_used[0] if step.neighbors_used else step.decision.retrieval.entries[0].name}"
        elif step.verdict == "compose":
            accuracy = _compose_accuracy(step.decision, models, eval_ldr)
            action_str = (f"compose top-{len(step.neighbors_used)}: "
                          f"{', '.join(step.neighbors_used)}")
        elif step.verdict == "spawn":
            # Accuracy of the spawn-with-context arm (run inside pipeline.step)
            spawned_model = pipeline.model_by_entry[step.new_entry.name]
            accuracy = evaluate_expert(spawned_model, eval_ldr, context=None)
            action_str = (f"spawn w/ context K={args.K} from: "
                          f"{', '.join(step.neighbors_used)}")
            print(f"    accuracy (spawn w/ context, N={max(args.budgets)}): {accuracy:.3f}")
            # Now run ablation
            print(f"\n    ── ABLATION on spawn path ──")
            # Remove the already-spawned entry to not pollute the registry
            registry.remove(step.new_entry.name)
            pipeline.model_by_entry.pop(step.new_entry.name, None)
            models.pop(step.new_entry.name, None)

            ablation_results = _ablation_for_spawn_query(
                q, registry, models,
                budgets=args.budgets, seeds=args.seeds,
                spec=context_spec, context_scale=args.context_scale,
                lr=args.lr, verbose=verbose,
            )
            ablation_by_query[q_name] = {
                arm: r.to_dict() for arm, r in ablation_results.items()
            }
            efficiency = ablation_results

        main_table_rows.append((
            q_name, step.verdict, action_str,
            accuracy if accuracy is not None else float("nan"),
            step.elapsed_s,
        ))
        step_dict = step.to_dict()
        step_dict["accuracy"] = accuracy
        step_dict["action_str"] = action_str
        if efficiency:
            step_dict["ablation"] = {
                arm: r.to_dict() for arm, r in efficiency.items()
            }
        steps_out.append(step_dict)

    # ── Reclustering finalize ──
    final_cluster = pipeline.finalize()
    if final_cluster is not None:
        print(f"\n  final reclustering: {final_cluster.n_clusters} clusters")

    # ── Render tables ──
    main_md = _render_main_table(main_table_rows)
    print("\n[4/4] RESULTS")
    print("\n" + main_md)

    ablation_md = ""
    if ablation_by_query:
        q_name = next(iter(ablation_by_query))
        print(f"\n  Sample-efficiency table for {q_name}:")
        ablation_results = {
            arm: SampleEfficiencyResult(
                arm=arm, budgets=d["budgets"],
                per_seed={int(s): {int(b): float(a) for b, a in inner.items()}
                          for s, inner in d["per_seed"].items()},
                mean_by_budget={int(b): float(v) for b, v in d["mean_by_budget"].items()},
                stdev_by_budget={int(b): float(v) for b, v in d["stdev_by_budget"].items()},
                raw=[],
            )
            for arm, d in ablation_by_query[q_name].items()
        }
        ablation_md = render_efficiency_table_md(list(ablation_results.values()))
        print("\n" + ablation_md)

    # ── Acceptance criterion ──
    acceptance = _check_acceptance(ablation_by_query, args.budgets[-1] if args.budgets else None)
    if acceptance is not None:
        print(f"\nAcceptance (B > A at N={acceptance['budget']}): "
              f"{'PASS' if acceptance['pass'] else 'FAIL'}  "
              f"mean_B={acceptance['mean_B']:.3f} "
              f"mean_A={acceptance['mean_A']:.3f} "
              f"std_A={acceptance['std_A']:.3f}")

    # ── Save ──
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode, "llm": args.llm,
        "config": {
            "K": args.K, "context_scale": args.context_scale,
            "match_threshold": args.match_threshold,
            "spawn_threshold": args.spawn_threshold,
            "seed_steps": seed_steps,
            "train_size": args.train_size, "batch_size": args.batch_size,
            "budgets": args.budgets, "seeds": args.seeds,
            "lr": args.lr,
        },
        "seed_experts": [
            {"name": f"{t}_seed{s}", "task": t, "task_labels": list(lbls), "seed": s}
            for t, lbls in SEED_TASKS.items() for s in SEEDS
        ],
        "queries": steps_out,
        "ablation": ablation_by_query,
        "acceptance": acceptance,
        "recluster_history": [
            {"n_clusters": r.n_clusters,
             "anchors": [sorted(a) for a in r.anchors]}
            for r in pipeline.cluster_history
        ],
        "tables": {"main_md": main_md, "ablation_md": ablation_md},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nsaved: {out_path}")

    if args.plot and ablation_by_query:
        q_name = next(iter(ablation_by_query))
        results_for_plot = {
            arm: SampleEfficiencyResult(
                arm=arm, budgets=d["budgets"],
                per_seed={int(s): {int(b): float(a) for b, a in inner.items()}
                          for s, inner in d["per_seed"].items()},
                mean_by_budget={int(b): float(v) for b, v in d["mean_by_budget"].items()},
                stdev_by_budget={int(b): float(v) for b, v in d["stdev_by_budget"].items()},
                raw=[],
            )
            for arm, d in ablation_by_query[q_name].items()
        }
        _plot_efficiency(results_for_plot, args.budgets, Path(args.plot_out))

    return 0


def _render_main_table(rows: list[tuple[str, str, str, float, float]]) -> str:
    header = ("| Query | Verdict | Action | Accuracy | Elapsed (s) |"
              "\n|---|---|---|---|---|")
    out = [header]
    for q, v, a, acc, t in rows:
        acc_s = f"{acc:.3f}" if np.isfinite(acc) else "—"
        out.append(f"| {q} | {v} | {a} | {acc_s} | {t:.2f} |")
    return "\n".join(out)


def _check_acceptance(ablation_by_query: dict, last_budget: int | None):
    """Return pass/fail for the acceptance criterion (B_mean > A_mean + A_std)."""
    if not ablation_by_query or last_budget is None:
        return None
    q = next(iter(ablation_by_query))
    d = ablation_by_query[q]
    try:
        A = d["A_no_context"]
        B = d["B_nearest_context"]
    except KeyError:
        return None
    mean_a = float(A["mean_by_budget"].get(str(last_budget),
                                             A["mean_by_budget"].get(last_budget, float("nan"))))
    std_a = float(A["stdev_by_budget"].get(str(last_budget),
                                             A["stdev_by_budget"].get(last_budget, float("nan"))))
    mean_b = float(B["mean_by_budget"].get(str(last_budget),
                                             B["mean_by_budget"].get(last_budget, float("nan"))))
    if not (np.isfinite(mean_a) and np.isfinite(mean_b) and np.isfinite(std_a)):
        return None
    return {
        "budget": last_budget,
        "mean_A": mean_a, "std_A": std_a, "mean_B": mean_b,
        "pass": bool(mean_b > mean_a + std_a),
    }


if __name__ == "__main__":
    raise SystemExit(main())
