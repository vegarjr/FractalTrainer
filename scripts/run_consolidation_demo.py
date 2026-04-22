"""v3 Sprint 17 — Direction G demo: consolidation / distillation.

Scenario: the registry has 15 specialist experts. Running every
query through 15 models is operationally awkward at scale. Can we
distill them into one generalist that handles most queries, with
fall-through to specialists only when the generalist is uncertain?

Metrics:
  1. Generalist accuracy on a held-out multi-task eval set (average
     of the registry's tasks).
  2. Fall-through rate: fraction of eval queries where the
     generalist's confidence < threshold, triggering specialist
     routing.
  3. Final accuracy of the consolidated router vs flat-always-
     specialist baseline.
  4. Latency: mean ms/query for generalist-first vs always-specialist.

Expected outcome: generalist handles ~70-90% of in-distribution
queries correctly; specialist fall-through catches the hard cases.
Net accuracy should be ≥ the per-task specialist's in-task accuracy.
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
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fractaltrainer.integration import (  # noqa: E402
    ConsolidatedRouter, train_generalist,
)
from fractaltrainer.integration.context_mlp import ContextAwareMLP  # noqa: E402

from run_fractal_demo import (  # noqa: E402
    SEED_TASKS, _build_seed_registry, _probe, _train_loader, _eval_loader,
    _probe_signature, _make_untrained_model,
)


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
    parser.add_argument("--distill-steps", type=int, default=500)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--n-eval-per-task", type=int, default=500)
    parser.add_argument("--latency-trials", type=int, default=100)
    parser.add_argument("--results-out",
                        default="results/consolidation_demo.json")
    args = parser.parse_args(argv)

    print("=" * 72)
    print(f"  CONSOLIDATION DEMO — {args.dataset} {args.arch} "
          f"(threshold={args.confidence_threshold})")
    print("=" * 72)

    # ── Build registry ──
    print("\n[1/5] training 15 seed experts...")
    probe = _probe(args.data_dir, args.n_probe, args.probe_seed,
                    dataset=args.dataset)
    registry, models = _build_seed_registry(
        probe, args.data_dir, args.seed_steps,
        args.train_size, args.batch_size, verbose=True,
        dataset=args.dataset, arch=args.arch,
    )
    print(f"  registry: {len(registry)} entries")

    # ── Build distillation loader (shared probe set from all tasks) ──
    print(f"\n[2/5] distilling into generalist "
          f"({args.distill_steps} steps, T={args.distill_temperature})...")
    specialists = list(models.values())

    # Distillation data: a broad probe drawn from the *full* MNIST
    # train distribution, without task-specific labels. The generalist
    # matches the specialists' teacher distribution.
    distill_loader = _distill_loader(
        args.train_size, args.batch_size, args.data_dir,
        args.probe_seed + 777, dataset=args.dataset,
    )
    generalist, stats = train_generalist(
        specialists, distill_loader,
        generalist_factory=lambda: _make_untrained_model(args.arch),
        n_steps=args.distill_steps,
        lr=0.01, temperature=args.distill_temperature,
        seed=42,
    )
    print(f"  distilled in {stats.elapsed_s:.1f}s, "
          f"final KL loss = {stats.final_loss:.4f}")

    # ── Evaluate generalist + consolidated router per task ──
    print("\n[3/5] per-task evaluation...")
    cr = ConsolidatedRouter(
        generalist, registry, models,
        confidence_threshold=args.confidence_threshold,
        n_classes=10,
    )

    per_task_results: list[dict] = []
    for task_name, target_labels in SEED_TASKS.items():
        eval_loader = list(_eval_loader(
            target_labels, args.n_eval_per_task, args.data_dir,
            seed=9000 + hash(task_name) % 1000,
            dataset=args.dataset,
        ))
        # Compute a single task-representative signature once — used
        # for specialist-lookup fallback. Use the task's first-seed
        # expert's signature.
        task_entry = None
        for entry in registry.entries():
            if entry.metadata.get("task") == task_name:
                task_entry = entry
                break
        q_sig = task_entry.signature if task_entry else None

        correct_g = correct_c = 0
        total = 0
        n_fell_through = 0
        for x, y in eval_loader:
            for i in range(x.size(0)):
                decision = cr.predict(x[i], query_signature=q_sig)
                # Convert binary y → class index (the specialists predict
                # binary, so class 1 = in-target-set)
                true_label = int(y[i].item())
                # Consolidated prediction: the specialist's output is
                # argmax over all N_classes logits. For a binary task,
                # class 1 = positive.
                # Compare directly against y (since specialists trained
                # on the binary relabeling).
                correct_c += int(decision.prediction == true_label)
                if decision.used_generalist:
                    correct_g += int(decision.prediction == true_label)
                else:
                    n_fell_through += 1
                total += 1
        acc_consolidated = correct_c / total
        acc_gen_only = correct_g / total  # count only generalist-answered queries
        fallthrough_rate = n_fell_through / total
        print(f"  {task_name:<20s} acc={acc_consolidated:.3f}  "
              f"gen-ans={acc_gen_only:.3f}  "
              f"fallthrough={fallthrough_rate:.2%}")
        per_task_results.append({
            "task": task_name,
            "accuracy_consolidated": acc_consolidated,
            "gen_answered_correct_rate": acc_gen_only,
            "fallthrough_rate": fallthrough_rate,
            "n_eval": total,
        })

    # ── Latency comparison ──
    print(f"\n[4/5] latency comparison ({args.latency_trials} trials)...")
    # Use one eval sample
    sample_task = next(iter(SEED_TASKS.keys()))
    sample_loader = _eval_loader(
        SEED_TASKS[sample_task], 1, args.data_dir,
        seed=42, dataset=args.dataset,
    )
    x_sample = next(iter(sample_loader))[0][:1]
    task_entry = next(e for e in registry.entries()
                       if e.metadata.get("task") == sample_task)
    q_sig = task_entry.signature

    # Always-specialist baseline: always route to nearest (simulate
    # what the pipeline does at match verdict)
    spec_model = models[task_entry.name]
    t0 = time.perf_counter()
    for _ in range(args.latency_trials):
        with torch.no_grad():
            logits = spec_model(x_sample, context=None)
            _ = F.softmax(logits, dim=1).argmax(dim=1)
    always_spec_dt = (time.perf_counter() - t0) / args.latency_trials

    t0 = time.perf_counter()
    for _ in range(args.latency_trials):
        _ = cr.predict(x_sample[0], query_signature=q_sig)
    consolidated_dt = (time.perf_counter() - t0) / args.latency_trials

    print(f"  always-specialist:    {always_spec_dt*1000:.3f} ms/query")
    print(f"  consolidated router:  {consolidated_dt*1000:.3f} ms/query")
    if consolidated_dt < always_spec_dt:
        print(f"  speedup (gen-first): "
              f"{always_spec_dt/consolidated_dt:.2f}×")
    else:
        print(f"  slowdown: {consolidated_dt/always_spec_dt:.2f}× "
              f"(generalist + confidence overhead)")

    # ── Save ──
    mean_acc = float(np.mean([r["accuracy_consolidated"]
                               for r in per_task_results]))
    mean_fallthrough = float(np.mean([r["fallthrough_rate"]
                                        for r in per_task_results]))
    out = {
        "config": vars(args),
        "n_specialists": len(specialists),
        "distill_stats": {
            "n_steps": stats.n_steps,
            "final_kl_loss": stats.final_loss,
            "elapsed_s": stats.elapsed_s,
        },
        "per_task": per_task_results,
        "mean_accuracy_consolidated": mean_acc,
        "mean_fallthrough_rate": mean_fallthrough,
        "latency": {
            "always_specialist_ms": always_spec_dt * 1000,
            "consolidated_ms": consolidated_dt * 1000,
        },
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[5/5] saved: {out_path}")
    print(f"\n  mean consolidated accuracy:  {mean_acc:.3f}")
    print(f"  mean fallthrough rate:       {mean_fallthrough:.2%}")
    return 0


def _distill_loader(n, batch_size, data_dir, seed, dataset):
    """Loader yielding (x, y) but y is ignored (generalist uses teacher)."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from run_fractal_demo import _dataset_cls, _transform
    cls, mean, std = _dataset_cls(dataset)
    base = cls(data_dir, train=True, download=True,
                transform=_transform(mean, std))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(base), size=n, replace=False)
    return DataLoader(Subset(base, idx.tolist()),
                       batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == "__main__":
    raise SystemExit(main())
