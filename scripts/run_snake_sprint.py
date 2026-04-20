"""v3 Sprint 13 — Snake teacher end-to-end driver.

Pipeline:
  1. Connect to local llama-server (Qwen2.5-Coder-7B).
  2. For each of K strategies in STRATEGY_PROMPTS, prompt the LLM,
     compile the function, play M games, collect demonstrations.
  3. Train K policy MLPs via behavior cloning.
  4. Register each as a FractalEntry with a probe-batch action-
     distribution signature.
  5. Evaluate:
       a. Each individual policy alone (50 games).
       b. Equal-weighted ensemble of all K policies (50 games).
       c. Top-1 routed ("best by signature centroid distance") —
          degenerate on this setup because there's only one task, but
          included for methodology parity with Sprint 9a.
       d. Random-3 ensemble baseline.
  6. Report + save JSON.

Output: results/snake_sprint.json with per-strategy train curves, demo
stats, game outcomes, and the comparison table.

Usage:
    python scripts/run_snake_sprint.py               # uses --llm local
    python scripts/run_snake_sprint.py --llm cli     # uses Claude CLI
    python scripts/run_snake_sprint.py --skip-llm    # reuse saved demos
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.repair.llm_client import (  # noqa: E402
    make_claude_cli_client,
    make_claude_client,
    make_local_llm_client,
)
from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402
from fractaltrainer.snake import (  # noqa: E402
    STRATEGY_PROMPTS,
    Demo,
    evaluate_ensemble,
    evaluate_single_policy,
    generate_demos_from_llm,
    probe_batch,
    train_policy,
)


def _get_llm(name: str, local_url: str):
    if name == "local":
        return make_local_llm_client(base_url=local_url, temperature=0.5,
                                       max_tokens=1024)
    if name == "cli":
        return make_claude_cli_client()
    if name == "api":
        return make_claude_client()
    raise ValueError(f"unknown --llm: {name!r}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="local",
                        choices=("local", "cli", "api"))
    parser.add_argument("--local-llm-url", type=str,
                        default="http://127.0.0.1:8080")
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--n-train-games", type=int, default=30)
    parser.add_argument("--n-eval-games", type=int, default=50)
    parser.add_argument("--n-probes", type=int, default=50)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--results-out", type=str,
                        default="results/snake_sprint.json")
    parser.add_argument("--demos-cache", type=str,
                        default="results/snake_demos.npz")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Reuse demos from --demos-cache instead of "
                             "calling the LLM again")
    args = parser.parse_args(argv)

    print(f"Sprint 13 — Snake Teacher")
    print(f"  LLM backend   : {args.llm}")
    print(f"  Board         : {args.height}×{args.width}")
    print(f"  Demos/strategy: {args.n_train_games} games")
    print(f"  Eval games    : {args.n_eval_games} per policy/ensemble")
    print(f"  Probe batch   : {args.n_probes} states (seed={args.probe_seed})")
    print()

    # ── 1. Generate / load demos ─────────────────────────────────
    demos_cache = Path(args.demos_cache)
    demos: dict[str, Demo] = {}
    if args.skip_llm and demos_cache.is_file():
        print(f"[1/5] Loading cached demos from {demos_cache}...")
        data = np.load(demos_cache, allow_pickle=True)
        for name in data.files:
            d = data[name].item()
            demos[name] = Demo(
                name=name, states=d["states"], actions=d["actions"],
                games_played=d["games_played"], errors=d["errors"],
                avg_survival=d["avg_survival"], avg_score=d["avg_score"],
                raw_code=d.get("raw_code", ""),
            )
        print(f"  Loaded {len(demos)} strategies.")
    else:
        print("[1/5] Generating demos via LLM...")
        llm_fn = _get_llm(args.llm, args.local_llm_url)
        demos = generate_demos_from_llm(
            llm_fn, STRATEGY_PROMPTS,
            n_games=args.n_train_games,
            height=args.height, width=args.width,
            verbose=True,
        )
        print(f"  Obtained {len(demos)} valid strategies "
              f"out of {len(STRATEGY_PROMPTS)} prompted.")
        # Cache for potential re-runs
        demos_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(demos_cache, **{
            name: np.array({
                "states": d.states, "actions": d.actions,
                "games_played": d.games_played, "errors": d.errors,
                "avg_survival": d.avg_survival, "avg_score": d.avg_score,
                "raw_code": d.raw_code,
            }, dtype=object)
            for name, d in demos.items()
        })
        print(f"  Cached to {demos_cache}")

    if not demos:
        print("ERROR: no valid strategies. Exiting.")
        return 1

    print()
    print("Strategy demo stats:")
    for name, d in demos.items():
        print(f"  {name:<16s}  {d.n_demos if hasattr(d, 'n_demos') else len(d.states):>5d} demos"
              f"  survived {d.avg_survival:5.1f} steps/game"
              f"  score {d.avg_score:4.2f}"
              f"  errors {d.errors}")

    # ── 2. Train policies ────────────────────────────────────────
    print()
    print("[2/5] Training policies via behavior cloning...")
    probe = probe_batch(n_probes=args.n_probes, height=args.height,
                         width=args.width, seed=args.probe_seed)
    policies = {}
    t_train_start = time.time()
    for name, d in demos.items():
        if len(d.states) == 0:
            print(f"  {name}: SKIP (no demos)")
            continue
        t0 = time.time()
        tp = train_policy(
            d.states, d.actions, name=name, probe=probe,
            n_epochs=args.n_epochs,
        )
        elapsed = time.time() - t0
        final_loss = tp.train_loss_curve[-1] if tp.train_loss_curve else float("nan")
        policies[name] = tp
        print(f"  {name:<16s}  final loss {final_loss:.3f}"
              f"  ({elapsed:.1f}s, {tp.n_demos} demos, sig dim={len(tp.signature)})")
    print(f"  Total training wall: {time.time() - t_train_start:.1f}s")

    # ── 3. Register in FractalRegistry ───────────────────────────
    print()
    print("[3/5] Registering in FractalRegistry...")
    registry = FractalRegistry()
    for name, tp in policies.items():
        entry = FractalEntry(
            name=name, signature=tp.signature,
            metadata={"task": name, "n_demos": tp.n_demos,
                      "board": f"{args.height}x{args.width}"},
        )
        registry.add(entry)
    print(f"  Registry size: {len(registry)}")
    # Pairwise signature distances (diagnostic)
    names_list = list(policies.keys())
    print(f"  Pairwise signature distances:")
    for i, na in enumerate(names_list):
        for nb in names_list[i + 1:]:
            da = policies[na].signature - policies[nb].signature
            d = float(np.linalg.norm(da))
            print(f"    {na} ↔ {nb}: {d:.3f}")

    # ── 4. Evaluate ──────────────────────────────────────────────
    print()
    print(f"[4/5] Evaluating policies on {args.n_eval_games} held-out games "
          f"(seed 5000+)...")
    results: dict[str, dict] = {}

    # 4a. Each individual policy
    for name, tp in policies.items():
        stats = evaluate_single_policy(
            tp.model, n_games=args.n_eval_games,
            height=args.height, width=args.width, seed_base=5000,
        )
        results[f"single:{name}"] = stats
        print(f"  single {name:<16s}  survival {stats['mean_survival']:6.2f}"
              f"  score {stats['mean_score']:.2f}"
              f"  max {stats['max_score']}")

    # 4b. Equal-weighted all-K ensemble
    models_all = [tp.model for tp in policies.values()]
    w_all = np.ones(len(models_all)) / len(models_all)
    stats_all = evaluate_ensemble(
        models_all, w_all, n_games=args.n_eval_games,
        height=args.height, width=args.width, seed_base=5000,
    )
    results["ensemble_all"] = stats_all
    print(f"  ensemble_all         survival {stats_all['mean_survival']:6.2f}"
          f"  score {stats_all['mean_score']:.2f}"
          f"  max {stats_all['max_score']}")

    # 4c. Random-3 ensemble (lower-bound control)
    rng = np.random.RandomState(2024)
    if len(models_all) >= 3:
        idx3 = rng.choice(len(models_all), size=3, replace=False)
        models3 = [models_all[i] for i in idx3]
        w3 = np.ones(3) / 3
        stats_r3 = evaluate_ensemble(
            models3, w3, n_games=args.n_eval_games,
            height=args.height, width=args.width, seed_base=5000,
        )
        results["ensemble_random3"] = stats_r3
        picked_names = [names_list[i] for i in idx3]
        print(f"  ensemble_random3     survival {stats_r3['mean_survival']:6.2f}"
              f"  score {stats_r3['mean_score']:.2f}  picks={picked_names}")

    # ── 5. Verdict ───────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    best_single = max(
        (k for k in results if k.startswith("single:")),
        key=lambda k: results[k]["mean_survival"],
    )
    best_survival_single = results[best_single]["mean_survival"]
    ens_survival = results["ensemble_all"]["mean_survival"]
    delta = ens_survival - best_survival_single
    print(f"  Best single policy     : {best_single}  "
          f"(survival {best_survival_single:.2f})")
    print(f"  All-K ensemble         : survival {ens_survival:.2f}")
    print(f"  Δ (ensemble − best)    : {delta:+.2f}")
    if delta > 1.0:
        verdict = "ENSEMBLE WINS (compose beats best single)"
    elif delta < -1.0:
        verdict = "BEST SINGLE WINS"
    else:
        verdict = "TIED within ±1 step"
    print(f"  Verdict: {verdict}")

    # ── Save ─────────────────────────────────────────────────────
    out = {
        "config": vars(args),
        "strategies": list(demos.keys()),
        "demo_stats": {n: {
            "n_demos": int(d.n_demos) if hasattr(d, "n_demos") else int(len(d.states)),
            "games_played": d.games_played,
            "errors": d.errors,
            "avg_survival": d.avg_survival,
            "avg_score": d.avg_score,
        } for n, d in demos.items()},
        "train_losses": {n: {
            "final": tp.train_loss_curve[-1] if tp.train_loss_curve else None,
            "curve_downsampled": tp.train_loss_curve[::20],
        } for n, tp in policies.items()},
        "pairwise_signature_distances": {
            f"{a}|{b}": float(np.linalg.norm(
                policies[a].signature - policies[b].signature))
            for i, a in enumerate(names_list)
            for b in names_list[i + 1:]
        },
        "eval_results": results,
        "verdict": verdict,
        "delta_ensemble_minus_best_single": delta,
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
