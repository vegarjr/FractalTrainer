"""ADAPTED from /home/vegar/Documents/Fractal/evolution/stage11_introspection/repair_loop.py
Source commit: ec2bddd2ea4cdf83693ecd1f9af84d4fadd24687
Original author: Vegar Ratdal (Fractal project)
License: MIT (same ownership)
Modifications:
  - Dropped DiagnosticReport, REPAIRABLE_ACTIONS, DATA_ACTIONS, and
    issue-extraction logic — not applicable outside Fractal's diagnostic
    pipeline.
  - Replaced _run_tests() with _run_training_probe() — retrains with the
    patched hparams and returns success/failure.
  - Replaced _measure_metric() with _measure_divergence() — runs the
    comparator and returns the divergence score.
  - Dropped PROTECTED_PREFIXES; the only patchable file is
    configs/hparams.yaml, enforced via allowed_files on the parser.
  - Added schema-gate validation (validate_hparams) before tests +
    divergence measurement.
  - repair() is the public entrypoint — takes (target, experiment_config,
    max_iters) rather than a diagnostic report.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from fractaltrainer.repair.context import (
    ALLOWED_FILES,
    GeometricContextGatherer,
    GeometricRepairContext,
)
from fractaltrainer.repair.hparam_config import load_hparams, validate_hparams
from fractaltrainer.repair.llm_client import make_claude_cli_client, make_claude_client
from fractaltrainer.repair.patch_parser import CodePatch, PatchParser
from fractaltrainer.repair.prompt_builder import PromptBuilder
from fractaltrainer.target.divergence import divergence_score, within_band
from fractaltrainer.target.target_shape import TargetShape


@dataclass
class RepairAttempt:
    iteration: int
    status: str                         # accepted, rolled_back, no_fix,
                                        # validation_failed, schema_failed,
                                        # no_improvement, error
    divergence_before: Optional[float]
    divergence_after: Optional[float]
    dim_before: Optional[float]
    dim_after: Optional[float]
    hparams_before: dict
    hparams_after: Optional[dict]
    patches: list[dict]
    elapsed_s: float
    summary: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "status": self.status,
            "divergence_before": self.divergence_before,
            "divergence_after": self.divergence_after,
            "dim_before": self.dim_before,
            "dim_after": self.dim_after,
            "hparams_before": self.hparams_before,
            "hparams_after": self.hparams_after,
            "patches": self.patches,
            "elapsed_s": round(self.elapsed_s, 2),
            "summary": self.summary,
            "error": self.error,
        }


def _mock_llm(system_prompt: str, user_prompt: str) -> str:
    return "NO_FIX_FOUND: mock LLM configured"


class RepairLoop:
    """LLM-driven fractal-shape-guided hyperparameter repair."""

    def __init__(
        self,
        project_root: str | Path,
        target: TargetShape,
        experiment_config: str | Path,
        hparams_path: str | Path = "configs/hparams.yaml",
        llm_fn: Callable[[str, str], str] | None = None,
        python_bin: str | None = None,
        log_path: str = "results/repair_log.jsonl",
        backup_dir: str = "results/repair_backups",
        probe_run_id: str | None = None,
    ):
        self.root = Path(project_root).resolve()
        self.target = target
        self.experiment_config = str(experiment_config)
        self.hparams_rel = str(hparams_path)
        self.hparams_abs = self.root / self.hparams_rel

        self.llm_fn = llm_fn or _mock_llm
        self.python_bin = python_bin or sys.executable

        self.log_path = self.root / log_path
        self.backup_dir = self.root / backup_dir
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.gatherer = GeometricContextGatherer(str(self.root))
        self.builder = PromptBuilder()
        self.parser = PatchParser()
        self.probe_run_id_prefix = probe_run_id or "probe"

    # ── Public API ────────────────────────────────────────────────

    def repair(self, max_iters: int | None = None,
               verbose: bool = False) -> list[RepairAttempt]:
        """Iteratively observe → compare → patch until within band or capped.

        Returns the list of RepairAttempt records.
        """
        max_iters = int(max_iters or self.target.max_repair_iters)
        attempts: list[RepairAttempt] = []

        for i in range(1, max_iters + 1):
            if verbose:
                print(f"\n[repair] iteration {i}/{max_iters}")

            attempt = self._one_iteration(i, verbose=verbose)
            attempts.append(attempt)
            self._append_log(attempt)

            if attempt.status == "accepted" and attempt.dim_after is not None \
                    and within_band(attempt.dim_after, self.target):
                if verbose:
                    print(f"[repair] within target band at iter {i} — halting")
                break
            if attempt.status == "no_fix":
                if verbose:
                    print(f"[repair] LLM reports no fix — halting")
                break

        return attempts

    # ── Single iteration ─────────────────────────────────────────

    def _one_iteration(self, iteration: int,
                       verbose: bool = False) -> RepairAttempt:
        t0 = time.time()

        hparams_before = load_hparams(self.hparams_abs)

        if verbose:
            print(f"[repair] probing with current hparams...")
        probe_before = self._run_training_probe(
            run_id=f"{self.probe_run_id_prefix}_{iteration}_before")
        if probe_before.get("error"):
            return RepairAttempt(
                iteration=iteration, status="error",
                divergence_before=None, divergence_after=None,
                dim_before=None, dim_after=None,
                hparams_before=hparams_before, hparams_after=None,
                patches=[], elapsed_s=time.time() - t0,
                error=f"training probe (before) failed: {probe_before['error']}",
            )

        report_before = self._run_comparator(probe_before["trajectory_path"])
        if report_before is None:
            return RepairAttempt(
                iteration=iteration, status="error",
                divergence_before=None, divergence_after=None,
                dim_before=None, dim_after=None,
                hparams_before=hparams_before, hparams_after=None,
                patches=[], elapsed_s=time.time() - t0,
                error="comparator (before) returned no report",
            )

        dim_before = float(
            report_before.get("primary_result", {}).get("dim", float("nan")))
        div_before = divergence_score(dim_before, self.target)

        if within_band(dim_before, self.target):
            return RepairAttempt(
                iteration=iteration, status="accepted",
                divergence_before=div_before, divergence_after=div_before,
                dim_before=dim_before, dim_after=dim_before,
                hparams_before=hparams_before, hparams_after=hparams_before,
                patches=[], elapsed_s=time.time() - t0,
                summary=f"already within band (dim={dim_before:.3f})",
            )

        context = self.gatherer.gather(
            comparison_report=report_before,
            target=self.target,
            log_path=str(self.log_path),
            hparams_path=self.hparams_rel,
        )

        prompt = self.builder.build(context)
        system_prompt = self.builder.SYSTEM_PROMPT.format(
            max_lines=context.max_lines_changed)

        try:
            response = self.llm_fn(system_prompt, prompt)
        except Exception as e:
            return RepairAttempt(
                iteration=iteration, status="error",
                divergence_before=div_before, divergence_after=None,
                dim_before=dim_before, dim_after=None,
                hparams_before=hparams_before, hparams_after=None,
                patches=[], elapsed_s=time.time() - t0,
                error=f"LLM call failed: {e}",
            )

        parse = self.parser.parse(response)
        if parse.no_fix:
            return RepairAttempt(
                iteration=iteration, status="no_fix",
                divergence_before=div_before, divergence_after=None,
                dim_before=dim_before, dim_after=None,
                hparams_before=hparams_before, hparams_after=None,
                patches=[], elapsed_s=time.time() - t0,
                summary=parse.no_fix_reason,
            )
        if parse.parse_error:
            return RepairAttempt(
                iteration=iteration, status="error",
                divergence_before=div_before, divergence_after=None,
                dim_before=dim_before, dim_after=None,
                hparams_before=hparams_before, hparams_after=None,
                patches=[p.to_dict() for p in parse.patches],
                elapsed_s=time.time() - t0,
                error=f"parse error: {parse.parse_error}",
            )

        validation_errors = self.parser.validate_patches(
            parse.patches, str(self.root),
            max_total_lines=context.max_lines_changed,
            protected_files=[],
            allowed_files=list(context.allowed_files),
        )
        if validation_errors:
            return RepairAttempt(
                iteration=iteration, status="validation_failed",
                divergence_before=div_before, divergence_after=None,
                dim_before=dim_before, dim_after=None,
                hparams_before=hparams_before, hparams_after=None,
                patches=[p.to_dict() for p in parse.patches],
                elapsed_s=time.time() - t0,
                error="; ".join(validation_errors),
            )

        backup_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        backed_up = self._backup_files(parse.patches, backup_id)

        try:
            self._apply_patches(parse.patches)
        except Exception as e:
            self._rollback(backed_up)
            return RepairAttempt(
                iteration=iteration, status="error",
                divergence_before=div_before, divergence_after=None,
                dim_before=dim_before, dim_after=None,
                hparams_before=hparams_before, hparams_after=None,
                patches=[p.to_dict() for p in parse.patches],
                elapsed_s=time.time() - t0,
                error=f"patch application failed: {e}",
            )

        # Schema gate — reject if post-patch hparams are invalid
        try:
            post_patch = load_hparams(self.hparams_abs)
        except Exception as e:
            self._rollback(backed_up)
            return RepairAttempt(
                iteration=iteration, status="schema_failed",
                divergence_before=div_before, divergence_after=None,
                dim_before=dim_before, dim_after=None,
                hparams_before=hparams_before, hparams_after=None,
                patches=[p.to_dict() for p in parse.patches],
                elapsed_s=time.time() - t0,
                error=f"post-patch yaml load failed: {e}",
            )
        schema_errors = validate_hparams(post_patch)
        if schema_errors:
            self._rollback(backed_up)
            return RepairAttempt(
                iteration=iteration, status="schema_failed",
                divergence_before=div_before, divergence_after=None,
                dim_before=dim_before, dim_after=None,
                hparams_before=hparams_before, hparams_after=None,
                patches=[p.to_dict() for p in parse.patches],
                elapsed_s=time.time() - t0,
                error="; ".join(schema_errors),
            )

        # Re-train + re-compare with patched hparams
        probe_after = self._run_training_probe(
            run_id=f"{self.probe_run_id_prefix}_{iteration}_after")
        if probe_after.get("error"):
            self._rollback(backed_up)
            return RepairAttempt(
                iteration=iteration, status="error",
                divergence_before=div_before, divergence_after=None,
                dim_before=dim_before, dim_after=None,
                hparams_before=hparams_before, hparams_after=post_patch,
                patches=[p.to_dict() for p in parse.patches],
                elapsed_s=time.time() - t0,
                error=f"training probe (after) failed: {probe_after['error']}",
            )

        report_after = self._run_comparator(probe_after["trajectory_path"])
        if report_after is None:
            self._rollback(backed_up)
            return RepairAttempt(
                iteration=iteration, status="error",
                divergence_before=div_before, divergence_after=None,
                dim_before=dim_before, dim_after=None,
                hparams_before=hparams_before, hparams_after=post_patch,
                patches=[p.to_dict() for p in parse.patches],
                elapsed_s=time.time() - t0,
                error="comparator (after) returned no report",
            )

        dim_after = float(
            report_after.get("primary_result", {}).get("dim", float("nan")))
        div_after = divergence_score(dim_after, self.target)

        # Outcome gate: accept only if divergence strictly decreased
        if div_after is None or div_before is None or div_after >= div_before:
            self._rollback(backed_up)
            return RepairAttempt(
                iteration=iteration, status="no_improvement",
                divergence_before=div_before, divergence_after=div_after,
                dim_before=dim_before, dim_after=dim_after,
                hparams_before=hparams_before, hparams_after=post_patch,
                patches=[p.to_dict() for p in parse.patches],
                elapsed_s=time.time() - t0,
                summary=(f"dim {dim_before:.3f} -> {dim_after:.3f}, "
                         f"div {div_before:.3f} -> {div_after:.3f} "
                         "(no improvement)"),
            )

        return RepairAttempt(
            iteration=iteration, status="accepted",
            divergence_before=div_before, divergence_after=div_after,
            dim_before=dim_before, dim_after=dim_after,
            hparams_before=hparams_before, hparams_after=post_patch,
            patches=[p.to_dict() for p in parse.patches],
            elapsed_s=time.time() - t0,
            summary=(f"dim {dim_before:.3f} -> {dim_after:.3f}, "
                     f"div {div_before:.3f} -> {div_after:.3f}"),
        )

    # ── File-level helpers ────────────────────────────────────────

    def _backup_files(self, patches: list[CodePatch], backup_id: str
                      ) -> dict[str, str]:
        backed = {}
        subdir = self.backup_dir / backup_id
        subdir.mkdir(parents=True, exist_ok=True)
        for patch in patches:
            src = self.root / patch.file_path
            if src.is_file():
                dst = subdir / patch.file_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                backed[patch.file_path] = str(dst)
        return backed

    def _apply_patches(self, patches: list[CodePatch]) -> None:
        for patch in patches:
            abs_path = self.root / patch.file_path
            content = abs_path.read_text()
            if patch.old_text not in content:
                raise ValueError(
                    f"old_text not found in {patch.file_path}")
            new_content = content.replace(patch.old_text, patch.new_text, 1)
            abs_path.write_text(new_content)

    def _rollback(self, backed_up: dict[str, str]) -> None:
        for rel_path, backup_path in backed_up.items():
            dst = self.root / rel_path
            try:
                shutil.copy2(backup_path, dst)
            except Exception:
                pass

    # ── Probe + measurement ──────────────────────────────────────

    def _run_training_probe(self, run_id: str) -> dict[str, Any]:
        """Run scripts/run_observation.py with the current hparams."""
        script = self.root / "scripts" / "run_observation.py"
        if not script.is_file():
            return {"error": f"observation script not found at {script}"}

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root / "src") + os.pathsep + env.get(
            "PYTHONPATH", "")

        try:
            result = subprocess.run(
                [self.python_bin, str(script),
                 "--config", self.experiment_config,
                 "--hparams", self.hparams_rel,
                 "--run-id", run_id],
                capture_output=True, text=True, timeout=600,
                cwd=str(self.root), env=env,
            )
        except subprocess.TimeoutExpired as e:
            return {"error": f"timeout: {e}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

        if result.returncode != 0:
            tail = (result.stderr or result.stdout)[-500:]
            return {"error": f"rc={result.returncode}: {tail}"}

        out_dir = self.root / "results" / "trajectories"
        traj = out_dir / f"{run_id}_trajectory.npy"
        if not traj.is_file():
            return {"error": f"no trajectory file at {traj}"}
        return {"trajectory_path": str(traj), "stdout": result.stdout}

    def _run_comparator(self, trajectory_path: str) -> dict | None:
        """Run scripts/run_comparison.py → return the loaded report dict."""
        script = self.root / "scripts" / "run_comparison.py"
        if not script.is_file():
            return None

        target_path = self.root / "configs" / "target_shape.yaml"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root / "src") + os.pathsep + env.get(
            "PYTHONPATH", "")
        try:
            result = subprocess.run(
                [self.python_bin, str(script),
                 "--trajectory", trajectory_path,
                 "--target", str(target_path)],
                capture_output=True, text=True, timeout=300,
                cwd=str(self.root), env=env,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        # Default report path = <trajectory>.comparison_report.json
        report_path = Path(trajectory_path).with_suffix(
            ".comparison_report.json")
        if not report_path.is_file():
            return None
        with open(report_path) as f:
            return json.load(f)

    # ── Logging ──────────────────────────────────────────────────

    def _append_log(self, attempt: RepairAttempt) -> None:
        entry = attempt.to_dict()
        entry["logged_at"] = datetime.now().isoformat()
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass
