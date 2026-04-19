"""FractalTrainer CLI — thin dispatcher over scripts/."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    # src/fractaltrainer/cli.py → repo root is ../../..
    return Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fractaltrainer",
                                     description="FractalTrainer CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("observe",
                   help="Instrument a training run; emit trajectory + metrics")
    sub.add_parser("compare",
                   help="Compare a trajectory to the target shape")
    sub.add_parser("repair",
                   help="Run the full closed loop")

    args, rest = parser.parse_known_args(argv)

    repo_root = _repo_root()
    script_map = {
        "observe": "scripts/run_observation.py",
        "compare": "scripts/run_comparison.py",
        "repair": "scripts/run_closed_loop.py",
    }
    script = repo_root / script_map[args.cmd]
    cmd = [sys.executable, str(script)] + rest
    env_prefix = f"PYTHONPATH={repo_root / 'src'}"
    print(f"[cli] {env_prefix} {' '.join(cmd)}")
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get(
        "PYTHONPATH", "")
    return subprocess.call(cmd, env=env, cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
