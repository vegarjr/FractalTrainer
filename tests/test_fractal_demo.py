"""Smoke tests for the Sprint 17 demo pipeline, end-to-end at tiny scale.

We test the pipeline at full-chain level with synthetic data (no
torchvision, no MNIST download) to keep CI fast and offline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _require_mnist_or_skip(data_dir: str):
    """Skip the real-data test if MNIST isn't already downloaded."""
    raw = Path(data_dir) / "MNIST" / "raw"
    if not raw.is_dir() or not any(raw.glob("*ubyte*")):
        pytest.skip(f"MNIST raw files not found under {raw}")


def test_demo_smoke_end_to_end_real_mnist():
    """Run the real demo script in smoke mode. Requires MNIST on disk."""
    data_dir = "results/data"
    # When running from tests/, cwd is FractalTrainer/, so the relative
    # path should resolve.
    repo_root = Path(__file__).resolve().parent.parent
    full_data_dir = repo_root / data_dir
    _require_mnist_or_skip(str(full_data_dir))

    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        import run_fractal_demo
    finally:
        sys.path.pop(0)

    out_json = repo_root / "results" / "fractal_demo_smoke_test.json"
    if out_json.exists():
        out_json.unlink()

    argv = [
        "--mode", "smoke",
        "--llm", "mock",
        "--data-dir", str(full_data_dir),
        "--results-out", str(out_json),
        "--seed-steps", "40",
        "--train-size", "512",
        "--budgets", "20", "40",
        "--n-probe", "30",
        "--n-eval", "200",
    ]
    rc = run_fractal_demo.main(argv)
    assert rc == 0
    assert out_json.is_file()
    with open(out_json) as f:
        payload = json.load(f)
    assert "queries" in payload
    assert "ablation" in payload
    assert "tables" in payload
    assert payload["tables"]["main_md"]  # non-empty
    # All three queries should appear
    q_names = {q["query_name"] for q in payload["queries"]}
    assert {"Q_match", "Q_compose", "Q_spawn"} <= q_names


def test_pipeline_api_surface_without_mnist():
    """Validate the public API surface without downloading MNIST."""
    from fractaltrainer.integration import (
        ContextAwareMLP, ContextSpec, FractalPipeline,
        MockDescriber, QueryInput, ReclusterPolicy,
        SampleEfficiencyResult, evaluate_expert,
        render_efficiency_table_md, sample_efficiency_curve,
        spawn_baseline, spawn_with_context, spawn_random_context,
    )
    # Smoke test — construct a pipeline with synthetic state
    from fractaltrainer.registry import FractalEntry, FractalRegistry

    reg = FractalRegistry()
    reg.add(FractalEntry(
        name="a", signature=np.array([0.0] + [0.0] * 99),
        metadata={"task": "a", "task_labels": [0, 1]}),
    )
    reg.add(FractalEntry(
        name="b", signature=np.array([10.0] + [0.0] * 99),
        metadata={"task": "b", "task_labels": [8, 9]}),
    )

    pipe = FractalPipeline(
        reg, MockDescriber(frozenset({0, 1})),
        match_threshold=1.0, spawn_threshold=2.0,
    )
    q = QueryInput(
        name="Q", pairs=[(0, 1), (1, 1)],
        truth_labels=frozenset({0, 1}),
    )
    sig = np.array([0.1] + [0.0] * 99)
    step = pipe.step(q, signature=sig)
    assert step.verdict == "match"
