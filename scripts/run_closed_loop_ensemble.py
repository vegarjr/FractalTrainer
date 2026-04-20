"""v3 Sprint 16b — Qwen ensemble wrapper: sample N times, return first
valid patch.

This wraps the llama-server client so each repair iteration becomes N
LLM calls at different temperatures. The first response that parses
as a valid patch is returned to the RepairLoop; outcome-gate-filtering
still happens at the loop level.

Why it might help: if Qwen's refusal rate is (say) 80% at any single
temperature, 5 independent draws drop the refusal rate to ~33%. The
loop then gets a chance to apply the patch and let the outcome gate
decide.

Usage:
    python scripts/run_closed_loop_ensemble.py --n-samples 5 --max-iters 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.repair.repair_loop import RepairLoop  # noqa: E402
from fractaltrainer.repair.llm_client import make_local_llm_client  # noqa: E402
from fractaltrainer.repair.patch_parser import PatchParser  # noqa: E402
from fractaltrainer.target.target_shape import load_target  # noqa: E402


def make_ensemble_llm(
    base_url: str, temperatures: list[float], *,
    max_tokens: int = 1024, verbose: bool = False,
) -> Callable[[str, str], str]:
    """Return a callable that, given (system, user), calls the server
    N times at N different temperatures and returns the FIRST response
    that parses as a valid patch. If none parse, return the first
    NO_FIX_FOUND response (or raw first if no NO_FIX either)."""
    parser = PatchParser()
    # Pre-create one client per temperature (pinging once)
    clients = []
    for t in temperatures:
        c = make_local_llm_client(base_url=base_url, temperature=t,
                                    max_tokens=max_tokens,
                                    ping_on_create=False)
        clients.append((t, c))

    def ensemble_call(system: str, user: str) -> str:
        first_response: str | None = None
        for t, client in clients:
            try:
                resp = client(system, user)
            except Exception as e:
                if verbose:
                    print(f"[ensemble] t={t} call failed: {e}")
                continue
            if first_response is None:
                first_response = resp
            # Check if parsable
            try:
                result = parser.parse(resp)
            except Exception:
                if verbose:
                    print(f"[ensemble] t={t}: unparseable")
                continue
            if result.no_fix:
                if verbose:
                    print(f"[ensemble] t={t}: NO_FIX_FOUND")
                continue
            if not result.patches:
                if verbose:
                    print(f"[ensemble] t={t}: no patches in response")
                continue
            # We got at least one patch — return this response, let
            # the RepairLoop's regular scope/schema/outcome gates
            # handle further validation.
            if verbose:
                print(f"[ensemble] t={t}: PATCH found, using this one")
            return resp
        # No valid patches — return whatever the first response was
        if verbose:
            print(f"[ensemble] no valid patches across {len(clients)} "
                  f"samples, returning first response")
        return first_response or "NO_FIX_FOUND: ensemble exhausted"

    return ensemble_call


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    parser.add_argument("--target", default="configs/target_shape.yaml")
    parser.add_argument("--hparams", default="configs/hparams.yaml")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.2, 0.5, 0.7, 0.9, 1.2])
    parser.add_argument("--local-llm-url", default="http://127.0.0.1:8080")
    parser.add_argument("--fewshot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results-out",
                        default="results/closed_loop_ensemble.json")
    args = parser.parse_args(argv)

    # Allow override of n-samples to slice temperatures
    temps = args.temperatures[: args.n_samples]
    print(f"Ensemble config: {len(temps)} samples at temperatures {temps}")
    print(f"Fewshot prompt: {args.fewshot}")

    llm_fn = make_ensemble_llm(args.local_llm_url, temps,
                                 verbose=args.verbose)

    target = load_target(args.target)
    loop = RepairLoop(
        project_root=str(REPO_ROOT),
        target=target,
        experiment_config=args.experiment,
        hparams_path=args.hparams,
        llm_fn=llm_fn,
        include_fewshot=args.fewshot,
    )
    t0 = time.time()
    attempts = loop.repair(max_iters=args.max_iters, verbose=True)
    elapsed = time.time() - t0

    print(f"\n====== ENSEMBLE CLOSED-LOOP SUMMARY ======")
    print(f"Wall clock: {elapsed:.1f}s")
    for a in attempts:
        print(f"iter {a.iteration}: status={a.status}  "
              f"dim {a.dim_before} → {a.dim_after}  "
              f"div {a.divergence_before} → {a.divergence_after}  "
              f"({a.elapsed_s:.1f}s)")
        if a.summary:
            print(f"    {a.summary}")

    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "temperatures": temps,
            "elapsed_s": elapsed,
            "attempts": [{
                "iteration": a.iteration, "status": a.status,
                "dim_before": a.dim_before, "dim_after": a.dim_after,
                "div_before": a.divergence_before,
                "div_after": a.divergence_after,
                "hparams_before": a.hparams_before,
                "hparams_after": a.hparams_after,
                "summary": a.summary,
                "elapsed_s": a.elapsed_s,
            } for a in attempts],
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
