"""v2 Sprint 6: multi-run closed-loop driver.

For each starting hparam configuration in a YAML file: write it to
configs/hparams.yaml, run the closed loop against the configured target,
and save the resulting repair log to results/multi_loop_logs/<name>.jsonl.
Then aggregate all logs into results/multi_loop_aggregated_repair_log.jsonl.

Feed the aggregated log into scripts/run_meta_observation.py (or any other
analysis) to study the shape of the repair loop's OWN sequence of fixes
across many starting points.

Usage:
    python scripts/run_multi_loop.py \\
        --starts configs/multi_loop_starting_points.yaml \\
        --target configs/target_shape_golden_opaque.yaml \\
        --llm cli \\
        --max-iters 3 \\
        [--python-bin <python>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def _save_hparams(hparams: dict, path: Path) -> None:
    ordered_keys = [
        "learning_rate", "batch_size", "weight_decay",
        "dropout", "init_seed", "optimizer",
    ]
    lines: list[str] = [
        "# Hyperparameters — written by scripts/run_multi_loop.py.",
        "# This file is overwritten on each run.",
        "",
    ]
    for k in ordered_keys:
        if k not in hparams:
            continue
        v = hparams[k]
        if isinstance(v, str):
            lines.append(f'{k}: "{v}"')
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n")


def _run_one_loop(
    name: str,
    hparams: dict,
    target: str,
    llm: str,
    max_iters: int,
    python_bin: str,
    hparams_path: Path,
    closed_loop_script: Path,
    repair_log_path: Path,
    per_run_log_dir: Path,
    verbose: bool,
) -> dict:
    """Set hparams, run the closed loop, capture the log for this run."""
    per_run_log_dir.mkdir(parents=True, exist_ok=True)

    # Reset hparams to this run's starting config
    _save_hparams(hparams, hparams_path)

    # Clear any prior repair log so this run's log is self-contained
    if repair_log_path.exists():
        repair_log_path.unlink()

    env_pythonpath = str(REPO_ROOT / "src")

    t0 = time.time()
    cmd = [
        python_bin, str(closed_loop_script),
        "--llm", llm,
        "--target", target,
        "--hparams", str(hparams_path.relative_to(REPO_ROOT)),
        "--max-iters", str(max_iters),
        "--python-bin", python_bin,
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = env_pythonpath + ":" + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
            cwd=str(REPO_ROOT), env=env,
        )
    except subprocess.TimeoutExpired as e:
        return {"name": name, "status": "timeout",
                "error": f"{e}", "elapsed_s": time.time() - t0}
    elapsed = time.time() - t0

    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-600:]
        return {
            "name": name, "status": "error",
            "error": f"rc={result.returncode} tail={tail}",
            "elapsed_s": elapsed,
        }

    # Capture the repair log this loop wrote
    per_run_log = per_run_log_dir / f"{name}.jsonl"
    if repair_log_path.exists():
        shutil.copy2(repair_log_path, per_run_log)
    else:
        per_run_log.write_text("")

    n_entries = 0
    if per_run_log.exists() and per_run_log.stat().st_size > 0:
        n_entries = sum(1 for line in per_run_log.read_text().splitlines()
                        if line.strip())

    if verbose:
        print(f"    log saved: {per_run_log}  entries={n_entries}  "
              f"elapsed={elapsed:.1f}s")

    return {
        "name": name, "status": "ok",
        "per_run_log": str(per_run_log),
        "n_log_entries": n_entries,
        "elapsed_s": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-run closed-loop driver")
    parser.add_argument("--starts", type=str,
                        default="configs/multi_loop_starting_points.yaml")
    parser.add_argument("--target", type=str,
                        default="configs/target_shape_golden_opaque.yaml")
    parser.add_argument("--hparams", type=str,
                        default="configs/hparams.yaml",
                        help="path the driver writes each starting config to")
    parser.add_argument("--llm", type=str, default="cli",
                        choices=("mock", "cli", "api"))
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--python-bin", type=str,
                        default=sys.executable,
                        help="python binary for the closed loop subprocess")
    parser.add_argument("--per-run-log-dir", type=str,
                        default="results/multi_loop_logs")
    parser.add_argument("--aggregated-log", type=str,
                        default="results/multi_loop_aggregated_repair_log.jsonl")
    parser.add_argument("--summary-json", type=str,
                        default="results/multi_loop_summary.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    with open(args.starts) as f:
        starts_cfg = yaml.safe_load(f)
    starts: list[dict] = starts_cfg.get("starts", [])
    if not starts:
        print(f"[multi] no starting configurations in {args.starts}",
              file=sys.stderr)
        return 1

    hparams_path = REPO_ROOT / args.hparams
    closed_loop_script = REPO_ROOT / "scripts" / "run_closed_loop.py"
    repair_log_path = REPO_ROOT / "results" / "repair_log.jsonl"
    per_run_log_dir = REPO_ROOT / args.per_run_log_dir
    aggregated_log_path = REPO_ROOT / args.aggregated_log
    summary_path = REPO_ROOT / args.summary_json

    print(f"[multi] {len(starts)} starting configurations")
    print(f"[multi] target: {args.target}")
    print(f"[multi] llm: {args.llm}  max_iters: {args.max_iters}")

    t_start = time.time()
    results: list[dict] = []
    for i, entry in enumerate(starts, start=1):
        name = entry["name"]
        hp = entry["hparams"]
        print(f"\n[multi] run {i}/{len(starts)}: {name}")
        print(f"        start: {hp}")
        r = _run_one_loop(
            name=name, hparams=hp,
            target=args.target, llm=args.llm,
            max_iters=args.max_iters,
            python_bin=args.python_bin,
            hparams_path=hparams_path,
            closed_loop_script=closed_loop_script,
            repair_log_path=repair_log_path,
            per_run_log_dir=per_run_log_dir,
            verbose=args.verbose,
        )
        print(f"        status: {r['status']}  "
              f"elapsed={r.get('elapsed_s', 0):.1f}s")
        if r.get("error"):
            print(f"        ERROR: {r['error']}")
        results.append(r)

    # Aggregate all per-run logs into one file (preserving per-entry run_id)
    aggregated_log_path.parent.mkdir(parents=True, exist_ok=True)
    n_aggregated = 0
    with open(aggregated_log_path, "w") as out:
        for r in results:
            per_run_log = r.get("per_run_log")
            if not per_run_log or not Path(per_run_log).exists():
                continue
            with open(per_run_log) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    entry["multi_loop_run_name"] = r["name"]
                    out.write(json.dumps(entry, default=str) + "\n")
                    n_aggregated += 1

    total_elapsed = time.time() - t_start
    summary = {
        "n_starts": len(starts),
        "n_successful_runs": sum(1 for r in results if r["status"] == "ok"),
        "n_errors": sum(1 for r in results if r["status"] != "ok"),
        "n_aggregated_log_entries": n_aggregated,
        "total_elapsed_s": total_elapsed,
        "aggregated_log": str(aggregated_log_path),
        "per_run_log_dir": str(per_run_log_dir),
        "target": args.target,
        "llm": args.llm,
        "max_iters": args.max_iters,
        "runs": results,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[multi] done in {total_elapsed:.1f}s")
    print(f"[multi] aggregated log: {aggregated_log_path}  "
          f"entries={n_aggregated}")
    print(f"[multi] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
