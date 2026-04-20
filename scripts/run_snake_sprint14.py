"""v3 Sprint 14 — Snake with LLM strategies as DIRECT experts.

Sprint 13 showed behavior-cloning LLM strategies into MLPs destroys
their generalization (cloned greedy_food dropped from 19.13 food/game
to 0.42). Sprint 14 tests the obvious fix: skip the MLP entirely and
use the LLM-generated Python functions as the registry entries.

Pipeline:
  1. Reuse the 5 strategies compiled in Sprint 13 (cached in
     results/snake_demos.npz — each has `raw_code` embedded).
  2. For each strategy, compute a signature by running the function
     on the canonical probe batch and collecting its action
     distribution. (Deterministic strategies produce one-hot
     distributions per state; fine — the signature still reflects
     strategy behavior.)
  3. Evaluate:
       a. Each strategy alone (50 games, seed 5000+)
       b. All-K majority-vote ensemble
       c. Random-3 majority-vote
       d. Signature-nearest-3 majority-vote (using each strategy's own
          signature as "query" — tests whether similar strategies make
          a weaker ensemble than diverse ones)
       e. Coverage-greedy-3 (holdout a seed-only mini-eval, greedy
          pick the K=3 with best ensemble survival on it)

Prediction: ensemble_all ≥ best_single, because error diversity across
actual algorithms (not chaotic clones) produces genuine complementarity
when majority-vote reconciles disagreements.

Usage:
    python scripts/run_snake_sprint14.py
    # Uses results/snake_demos.npz for compiled strategy source code.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.snake.env import (  # noqa: E402
    ACTIONS,
    ACTION_NAMES,
    SnakeEnv,
    probe_batch,
)
from fractaltrainer.snake.teacher import _compile_strategy  # noqa: E402


# ── Action normalization ────────────────────────────────────────────

def _normalize_action(raw) -> int | None:
    if isinstance(raw, str):
        raw = raw.strip().lower()
        rev = {v: k for k, v in ACTION_NAMES.items()}
        return rev.get(raw)
    if isinstance(raw, int) and 0 <= raw < 4:
        return raw
    return None


# ── Signature from a strategy's action distribution ─────────────────

def strategy_signature(strategy_fn: Callable, probe: np.ndarray,
                        height: int, width: int) -> np.ndarray:
    """For each probe state, call strategy_fn and record the action.
    Return a (n_probes * 4,) one-hot flattened signature."""
    n = probe.shape[0]
    sig = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        board_flat = probe[i].reshape(height, width)
        # Derive snake + food from the flat state (hacky but works)
        snake_cells = [(r, c) for r in range(height) for c in range(width)
                        if board_flat[r, c] in (1, 2)]
        head_cells = [(r, c) for r in range(height) for c in range(width)
                       if board_flat[r, c] == 2]
        food_cells = [(r, c) for r in range(height) for c in range(width)
                       if board_flat[r, c] == 3]
        if not head_cells or not food_cells:
            # Degenerate probe — spread prob uniformly
            sig[i] = 0.25
            continue
        head = head_cells[0]
        # Rough snake ordering: head first, then body in any order
        body = [c for c in snake_cells if c != head]
        snake = [head] + body
        food = food_cells[0]
        board = board_flat.tolist()
        try:
            action = _normalize_action(strategy_fn(board, snake, food))
            if action is None:
                sig[i] = 0.25
            else:
                sig[i, action] = 1.0
        except Exception:
            sig[i] = 0.25
    return sig.flatten()


# ── Game playing via strategies ─────────────────────────────────────

def play_game_with_strategy(strategy_fn: Callable, seed: int,
                              height: int, width: int) -> tuple[int, int]:
    """Play one game with the given strategy. Return (steps, score)."""
    env = SnakeEnv(height=height, width=width, seed=seed)
    env.reset()
    while not env.done:
        board = env.render_board().tolist()
        snake = list(env.snake)
        food = env.food
        try:
            raw = strategy_fn(board, snake, food)
            action = _normalize_action(raw)
            if action is None:
                action = 0  # fallback
        except Exception:
            action = 0
        env.step(action)
    return env.steps, env.score


def play_game_with_ensemble(strategy_fns: list[Callable], seed: int,
                              height: int, width: int) -> tuple[int, int]:
    """Play with majority vote across K strategy functions. Ties
    broken by the first action in {0,1,2,3} that got the max count."""
    env = SnakeEnv(height=height, width=width, seed=seed)
    env.reset()
    while not env.done:
        board = env.render_board().tolist()
        snake = list(env.snake)
        food = env.food
        votes = []
        for fn in strategy_fns:
            try:
                raw = fn(board, snake, food)
                a = _normalize_action(raw)
                if a is not None:
                    votes.append(a)
            except Exception:
                pass
        if not votes:
            action = 0
        else:
            counts = Counter(votes)
            max_votes = max(counts.values())
            tied = sorted([a for a, c in counts.items() if c == max_votes])
            action = tied[0]
        env.step(action)
    return env.steps, env.score


def evaluate_strategy(strategy_fn: Callable, n_games: int,
                       height: int, width: int, seed_base: int) -> dict:
    survs, scores = [], []
    for g in range(n_games):
        s, sc = play_game_with_strategy(strategy_fn, seed_base + g,
                                           height, width)
        survs.append(s); scores.append(sc)
    return {
        "n_games": n_games,
        "mean_survival": float(np.mean(survs)),
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
    }


def evaluate_ensemble(strategy_fns: list[Callable], n_games: int,
                       height: int, width: int, seed_base: int) -> dict:
    survs, scores = [], []
    for g in range(n_games):
        s, sc = play_game_with_ensemble(strategy_fns, seed_base + g,
                                          height, width)
        survs.append(s); scores.append(sc)
    return {
        "n_games": n_games,
        "mean_survival": float(np.mean(survs)),
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
    }


# ── Main ────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos-cache", type=str,
                        default="results/snake_demos.npz",
                        help="Sprint 13 cache with raw_code per strategy")
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--n-eval-games", type=int, default=50)
    parser.add_argument("--n-probes", type=int, default=50)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--results-out", type=str,
                        default="results/snake_sprint14.json")
    args = parser.parse_args(argv)

    print("Sprint 14 — LLM strategies as direct experts")
    print(f"  Board          : {args.height}×{args.width}")
    print(f"  Eval games     : {args.n_eval_games} per condition (seed 5000+)")
    print(f"  Demo cache     : {args.demos_cache}")
    print()

    # ── 1. Load Sprint 13 strategies ─────────────────────────────
    cache = Path(args.demos_cache)
    if not cache.is_file():
        print(f"ERROR: {cache} not found. Run Sprint 13 first.")
        return 1
    data = np.load(cache, allow_pickle=True)
    strategies: dict[str, Callable] = {}
    raw_codes: dict[str, str] = {}
    for name in data.files:
        d = data[name].item()
        code = d.get("raw_code", "")
        if not code:
            print(f"  {name}: NO raw_code in cache — skipping")
            continue
        try:
            fn = _compile_strategy(code)
            strategies[name] = fn
            raw_codes[name] = code
        except Exception as e:
            print(f"  {name}: compile failed ({e}) — skipping")
    print(f"Loaded {len(strategies)} compiled strategies.")

    if len(strategies) < 2:
        print("ERROR: need at least 2 strategies for ensemble tests.")
        return 1

    # ── 2. Compute signatures ────────────────────────────────────
    print(f"\nComputing signatures on {args.n_probes} probe states...")
    probe = probe_batch(n_probes=args.n_probes, height=args.height,
                         width=args.width, seed=args.probe_seed)
    signatures: dict[str, np.ndarray] = {}
    for name, fn in strategies.items():
        sig = strategy_signature(fn, probe, args.height, args.width)
        signatures[name] = sig
        print(f"  {name:<16s}  sig dim={len(sig)}  "
              f"norm={np.linalg.norm(sig):.3f}")

    # Pairwise distances
    names = list(strategies.keys())
    print("\nPairwise signature distances:")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(np.linalg.norm(signatures[a] - signatures[b]))
            print(f"  {a} ↔ {b}: {d:.3f}")

    # ── 3. Evaluate each strategy alone ──────────────────────────
    print(f"\n[1/5] Single-strategy eval ({args.n_eval_games} games each)...")
    single_results: dict[str, dict] = {}
    for name, fn in strategies.items():
        t0 = time.time()
        stats = evaluate_strategy(
            fn, args.n_eval_games, args.height, args.width,
            seed_base=5000)
        single_results[name] = stats
        print(f"  {name:<16s}  surv {stats['mean_survival']:6.2f}  "
              f"score {stats['mean_score']:5.2f}  "
              f"max {stats['max_score']:>2}  "
              f"({time.time()-t0:.1f}s)")

    # ── 4. All-K majority vote ───────────────────────────────────
    print(f"\n[2/5] All-K ({len(strategies)}) majority-vote ensemble...")
    fns_all = list(strategies.values())
    stats_all = evaluate_ensemble(fns_all, args.n_eval_games,
                                    args.height, args.width,
                                    seed_base=5000)
    print(f"  all-K ensemble      surv {stats_all['mean_survival']:6.2f}  "
          f"score {stats_all['mean_score']:5.2f}  "
          f"max {stats_all['max_score']}")

    # ── 5. Random-3 (fixed seed) ─────────────────────────────────
    print("\n[3/5] Random-3 majority-vote ensemble...")
    rng = np.random.RandomState(2024)
    if len(strategies) >= 3:
        idx3 = rng.choice(len(strategies), size=3, replace=False)
        picks_r3 = [names[i] for i in idx3]
        fns_r3 = [strategies[n] for n in picks_r3]
        stats_r3 = evaluate_ensemble(fns_r3, args.n_eval_games,
                                       args.height, args.width,
                                       seed_base=5000)
        print(f"  random-3            surv {stats_r3['mean_survival']:6.2f}  "
              f"score {stats_r3['mean_score']:5.2f}  "
              f"max {stats_r3['max_score']}  picks={picks_r3}")
    else:
        stats_r3 = None

    # ── 6. Signature-nearest-3 to "median" strategy ──────────────
    print("\n[4/5] Signature-nearest-3 to registry centroid...")
    centroid = np.mean([signatures[n] for n in names], axis=0)
    dists = [(float(np.linalg.norm(signatures[n] - centroid)), n)
             for n in names]
    dists.sort()
    picks_near3 = [n for _, n in dists[:3]]
    fns_near3 = [strategies[n] for n in picks_near3]
    stats_near3 = evaluate_ensemble(fns_near3, args.n_eval_games,
                                      args.height, args.width,
                                      seed_base=5000)
    print(f"  nearest-3           surv {stats_near3['mean_survival']:6.2f}  "
          f"score {stats_near3['mean_score']:5.2f}  "
          f"max {stats_near3['max_score']}  picks={picks_near3}")

    # ── 7. Coverage-greedy-3 on seed-4000 mini-eval ──────────────
    print("\n[5/5] Coverage-greedy-3 (selected on held-out mini-games)...")
    # Use 10 games from seed 4000-4009 as the selection set; each
    # triple candidate is evaluated on its survival, pick the best.
    # C(5, 3) = 10 triples — brute force is fine.
    mini_n = 10
    best_tri = None; best_surv = -1
    for tri in itertools.combinations(names, 3):
        fns_t = [strategies[n] for n in tri]
        survs = []
        for g in range(mini_n):
            s, _ = play_game_with_ensemble(fns_t, 4000 + g,
                                             args.height, args.width)
            survs.append(s)
        m = float(np.mean(survs))
        if m > best_surv:
            best_surv = m; best_tri = tri
    fns_cov3 = [strategies[n] for n in best_tri]
    stats_cov3 = evaluate_ensemble(fns_cov3, args.n_eval_games,
                                     args.height, args.width,
                                     seed_base=5000)
    print(f"  coverage-greedy-3   surv {stats_cov3['mean_survival']:6.2f}  "
          f"score {stats_cov3['mean_score']:5.2f}  "
          f"max {stats_cov3['max_score']}  picks={list(best_tri)}")
    print(f"  (selected on mini-eval 4000-4009, survival {best_surv:.2f})")

    # ── Verdict ──────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    best_single_name = max(single_results,
                            key=lambda n: single_results[n]["mean_survival"])
    best_single_surv = single_results[best_single_name]["mean_survival"]
    best_single_score = single_results[best_single_name]["mean_score"]

    conditions = {
        "best_single": (best_single_surv, best_single_score,
                         f"(={best_single_name})"),
        "all_K": (stats_all["mean_survival"], stats_all["mean_score"], ""),
        "nearest_3": (stats_near3["mean_survival"],
                        stats_near3["mean_score"],
                        f"(={picks_near3})"),
        "coverage_3": (stats_cov3["mean_survival"],
                         stats_cov3["mean_score"],
                         f"(={list(best_tri)})"),
    }
    if stats_r3 is not None:
        conditions["random_3"] = (stats_r3["mean_survival"],
                                    stats_r3["mean_score"],
                                    f"(={picks_r3})")

    print(f"  {'condition':<16s}  {'survival':>9s}  {'score':>6s}  details")
    for c, (surv, sc, detail) in conditions.items():
        print(f"  {c:<16s}  {surv:>9.2f}  {sc:>6.2f}  {detail}")

    surv_all = stats_all["mean_survival"]
    surv_cov = stats_cov3["mean_survival"]
    d_all = surv_all - best_single_surv
    d_cov = surv_cov - best_single_surv

    if d_cov > 1.0 or d_all > 1.0:
        verdict = "ENSEMBLE WINS"
    elif d_cov < -1.0 and d_all < -1.0:
        verdict = "BEST SINGLE WINS"
    else:
        verdict = "TIED"
    print(f"\n  Δ(all-K − best)       = {d_all:+.2f}")
    print(f"  Δ(coverage-3 − best)  = {d_cov:+.2f}")
    print(f"  VERDICT: {verdict}")

    # ── Save ─────────────────────────────────────────────────────
    out = {
        "config": vars(args),
        "strategies": list(strategies.keys()),
        "signatures_norms": {n: float(np.linalg.norm(signatures[n]))
                              for n in names},
        "pairwise_signature_distances": {
            f"{a}|{b}": float(np.linalg.norm(signatures[a] - signatures[b]))
            for i, a in enumerate(names) for b in names[i + 1:]
        },
        "single_results": single_results,
        "all_K_ensemble": stats_all,
        "nearest3_ensemble": {"picks": picks_near3, **stats_near3},
        "coverage3_ensemble": {"picks": list(best_tri),
                                "selection_survival": best_surv,
                                **stats_cov3},
        "random3_ensemble": (
            {"picks": picks_r3, **stats_r3}
            if stats_r3 is not None else None),
        "best_single_name": best_single_name,
        "delta_all_minus_best": d_all,
        "delta_coverage_minus_best": d_cov,
        "verdict": verdict,
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
