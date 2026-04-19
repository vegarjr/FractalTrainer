"""Sprint 3 demo: full observe → compare → patch → retrain → measure loop.

Usage:
    python scripts/run_closed_loop.py --experiment configs/experiment.yaml \
        --target configs/target_shape.yaml --max-iters 5 --llm mock

    --llm mock : always returns NO_FIX_FOUND (safe, used in CI)
    --llm cli  : calls `claude --print` (Max subscription, no API cost)
    --llm api  : calls Anthropic API (requires ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.repair.repair_loop import RepairLoop  # noqa: E402
from fractaltrainer.repair.llm_client import (  # noqa: E402
    make_claude_cli_client,
    make_claude_client,
)
from fractaltrainer.target.target_shape import load_target  # noqa: E402


def _make_llm_fn(name: str):
    if name == "mock":
        return None  # RepairLoop's default _mock_llm
    if name == "cli":
        return make_claude_cli_client()
    if name == "api":
        return make_claude_client()
    raise ValueError(f"unknown --llm: {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fractal-shape-guided closed loop")
    parser.add_argument("--experiment", type=str, default="configs/experiment.yaml")
    parser.add_argument("--target", type=str, default="configs/target_shape.yaml")
    parser.add_argument("--hparams", type=str, default="configs/hparams.yaml")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="override target.max_repair_iters")
    parser.add_argument("--llm", type=str, default="mock",
                        choices=("mock", "cli", "api"))
    parser.add_argument("--python-bin", type=str, default=None,
                        help="python binary to use for training probes "
                        "(default: current sys.executable)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    target = load_target(args.target)
    llm_fn = _make_llm_fn(args.llm)

    loop = RepairLoop(
        project_root=str(REPO_ROOT),
        target=target,
        experiment_config=args.experiment,
        hparams_path=args.hparams,
        llm_fn=llm_fn,
        python_bin=args.python_bin,
    )

    attempts = loop.repair(max_iters=args.max_iters, verbose=args.verbose)

    print("\n====== CLOSED-LOOP SUMMARY ======")
    for a in attempts:
        print(f"iter {a.iteration}: status={a.status}  "
              f"dim {a.dim_before} → {a.dim_after}  "
              f"div {a.divergence_before} → {a.divergence_after}  "
              f"(elapsed {a.elapsed_s:.1f}s)")
        if a.error:
            print(f"    error: {a.error}")
        if a.summary:
            print(f"    {a.summary}")

    out_path = REPO_ROOT / "results" / "closed_loop_summary.json"
    with open(out_path, "w") as f:
        json.dump([a.to_dict() for a in attempts], f, indent=2, default=str)
    print(f"\nsummary saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
