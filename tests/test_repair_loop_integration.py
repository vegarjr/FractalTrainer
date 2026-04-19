"""Integration test for RepairLoop orchestration — mocks the subprocess
training/comparator calls so the test is fast and deterministic.

Covers:
  - Context → prompt → parse → validate → apply → measure → accept
  - Divergence improvement → accept path
  - Divergence worsens → no_improvement rollback path
  - Schema-invalid patch → schema_failed rollback path
  - NO_FIX_FOUND response → no_fix halt
  - Already within band at iteration start → accepted (no patch)
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from fractaltrainer.repair.repair_loop import RepairLoop
from fractaltrainer.target.target_shape import TargetShape, ProjectionSpec


VALID_HPARAMS_YAML = """learning_rate: 0.1
batch_size: 64
weight_decay: 0.0
dropout: 0.0
init_seed: 42
optimizer: "sgd"
"""

VALID_EXPERIMENT_YAML = """model:
  name: "mlp"
  hidden_dims: [32]
  input_dim: 4
  output_dim: 2
  activation: "relu"

dataset:
  name: "synthetic"
  train_subset_size: 64
  test_subset_size: 16
  data_dir: "results/data"

training:
  n_steps: 10
  snapshot_every: 2

output:
  out_dir: "results/trajectories"

hparams_path: "configs/hparams.yaml"
"""


def _make_target() -> TargetShape:
    return TargetShape(
        dim_target=1.5,
        tolerance=0.3,
        method="correlation_dim",
        hysteresis=1.2,
        max_repair_iters=3,
        projection=ProjectionSpec(method="random_proj", n_components=8, seed=0),
    )


def _make_project(tmp: Path) -> Path:
    (tmp / "configs").mkdir()
    (tmp / "configs" / "hparams.yaml").write_text(VALID_HPARAMS_YAML)
    (tmp / "configs" / "experiment.yaml").write_text(VALID_EXPERIMENT_YAML)
    (tmp / "configs" / "target_shape.yaml").write_text(
        "dim_target: 1.5\ntolerance: 0.3\nmethod: correlation_dim\n"
        "hysteresis: 1.2\nmax_repair_iters: 3\n"
        "projection:\n  method: random_proj\n  n_components: 8\n  seed: 0\n"
    )
    (tmp / "scripts").mkdir()
    # Lightweight stub scripts — repair_loop.py checks file existence before
    # invoking them. The real subprocesses are replaced by patched methods.
    (tmp / "scripts" / "run_observation.py").write_text("# stub\n")
    (tmp / "scripts" / "run_comparison.py").write_text("# stub\n")
    (tmp / "results").mkdir()
    (tmp / "results" / "trajectories").mkdir()
    return tmp


class _ScriptedLoop(RepairLoop):
    """RepairLoop subclass that fakes the heavy subprocess calls."""

    def __init__(self, *args, scripted_dims: list[float], **kwargs):
        super().__init__(*args, **kwargs)
        self._scripted_dims = list(scripted_dims)
        self._call_counter = 0

    def _run_training_probe(self, run_id: str) -> dict:
        # Write a stub trajectory file for the comparator stub to "find"
        out = self.root / "results" / "trajectories" / f"{run_id}_trajectory.npy"
        np.save(out, np.zeros((25, 8)))
        return {"trajectory_path": str(out)}

    def _run_comparator(self, trajectory_path: str) -> dict | None:
        if not self._scripted_dims:
            return None
        dim = self._scripted_dims.pop(0)
        return {
            "trajectory_path": trajectory_path,
            "trajectory_shape": [25, 1000],
            "projection": {
                "method": "random_proj", "n_components": 8, "seed": 0,
                "projected_shape": [25, 8],
            },
            "target": self.target.to_dict(),
            "primary_result": {"method": "correlation_dim", "dim": dim,
                               "r_squared": 0.99},
            "sanity_box_counting_3d": {"dim": dim},
            "divergence": {
                "score": abs(dim - self.target.dim_target) / self.target.tolerance,
                "within_band": (self.target.band_low <= dim
                                <= self.target.band_high),
                "should_intervene": (
                    abs(dim - self.target.dim_target) / self.target.tolerance
                    > self.target.hysteresis),
                "band": [self.target.band_low, self.target.band_high],
            },
            "baseline_metrics": {
                "trajectory_metrics": {"total_path_length": 10.0},
                "fractal_summary": {"compression_ratio": 0.5},
            },
        }


def _llm_patch_lr(new_lr: float):
    def _fn(system_prompt: str, user_prompt: str) -> str:
        return f"""<<<PATCH
file: configs/hparams.yaml
---old---
learning_rate: 0.1
---new---
learning_rate: {new_lr}
>>>PATCH"""
    return _fn


def _llm_no_fix(reason: str):
    return lambda sp, up: f"NO_FIX_FOUND: {reason}"


def test_accept_when_divergence_improves():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_project(Path(tmp))
        loop = _ScriptedLoop(
            project_root=root,
            target=_make_target(),
            experiment_config="configs/experiment.yaml",
            llm_fn=_llm_patch_lr(0.05),
            # Sequence: 2.5 (before iter 1) → 1.45 (after iter 1, inside band)
            scripted_dims=[2.5, 1.45],
        )
        attempts = loop.repair(max_iters=1)
        assert len(attempts) == 1
        a = attempts[0]
        assert a.status == "accepted", f"unexpected status: {a.status} err={a.error}"
        assert a.dim_before == pytest.approx(2.5)
        assert a.dim_after == pytest.approx(1.45)
        # hparams file should now show the new lr
        new_hparams = yaml.safe_load(
            (root / "configs/hparams.yaml").read_text())
        assert new_hparams["learning_rate"] == pytest.approx(0.05)


def test_rollback_when_divergence_worsens():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_project(Path(tmp))
        loop = _ScriptedLoop(
            project_root=root,
            target=_make_target(),
            experiment_config="configs/experiment.yaml",
            llm_fn=_llm_patch_lr(0.08),
            # 2.5 (before) → 3.0 (after) — worse
            scripted_dims=[2.5, 3.0],
        )
        attempts = loop.repair(max_iters=1)
        assert len(attempts) == 1
        a = attempts[0]
        assert a.status == "no_improvement"
        # hparams file should be back to 0.1
        restored = yaml.safe_load(
            (root / "configs/hparams.yaml").read_text())
        assert restored["learning_rate"] == pytest.approx(0.1)


def test_no_fix_halts_immediately():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_project(Path(tmp))
        loop = _ScriptedLoop(
            project_root=root,
            target=_make_target(),
            experiment_config="configs/experiment.yaml",
            llm_fn=_llm_no_fix("target seems unreachable"),
            scripted_dims=[2.5],  # only one probe needed before LLM short-circuit
        )
        attempts = loop.repair(max_iters=3)
        # Should halt after first no_fix; only 1 attempt recorded
        assert len(attempts) == 1
        assert attempts[0].status == "no_fix"


def test_schema_gate_rejects_out_of_range():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_project(Path(tmp))
        loop = _ScriptedLoop(
            project_root=root,
            target=_make_target(),
            experiment_config="configs/experiment.yaml",
            # 100.0 is out of range; schema gate must reject
            llm_fn=_llm_patch_lr(100.0),
            scripted_dims=[2.5],
        )
        attempts = loop.repair(max_iters=1)
        assert len(attempts) == 1
        assert attempts[0].status == "schema_failed"
        restored = yaml.safe_load(
            (root / "configs/hparams.yaml").read_text())
        assert restored["learning_rate"] == pytest.approx(0.1)


def test_already_in_band_accepts_without_patching():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_project(Path(tmp))
        loop = _ScriptedLoop(
            project_root=root,
            target=_make_target(),
            experiment_config="configs/experiment.yaml",
            llm_fn=_llm_patch_lr(0.01),
            scripted_dims=[1.5],  # dead center of band
        )
        attempts = loop.repair(max_iters=3)
        assert len(attempts) == 1
        assert attempts[0].status == "accepted"
        assert attempts[0].dim_before == pytest.approx(1.5)
        assert attempts[0].dim_after == pytest.approx(1.5)
        assert attempts[0].patches == []


def test_log_file_appended_each_iteration():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_project(Path(tmp))
        loop = _ScriptedLoop(
            project_root=root,
            target=_make_target(),
            experiment_config="configs/experiment.yaml",
            llm_fn=_llm_no_fix("done"),
            scripted_dims=[2.5],
        )
        loop.repair(max_iters=1)
        log_path = root / "results" / "repair_log.jsonl"
        assert log_path.exists()
        lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["status"] == "no_fix"
        assert "logged_at" in parsed
