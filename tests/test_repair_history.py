"""Tests for RepairHistoryReader and embed_hparams."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fractaltrainer.observer.repair_history import (
    MetaTrajectory,
    OPTIMIZER_INDEX,
    RepairHistoryReader,
    embed_hparams,
)


def _valid_hparams(**overrides) -> dict:
    base = {
        "learning_rate": 0.01,
        "batch_size": 64,
        "weight_decay": 0.0001,
        "dropout": 0.1,
        "init_seed": 42,
        "optimizer": "adam",
    }
    base.update(overrides)
    return base


def test_embed_hparams_shape_and_logs():
    vec = embed_hparams(_valid_hparams(learning_rate=0.001, batch_size=32))
    assert vec.shape == (6,)
    assert vec[0] == pytest.approx(-3.0)  # log10(1e-3)
    assert vec[1] == pytest.approx(5.0)   # log2(32)


def test_embed_hparams_optimizer_index():
    for name, idx in OPTIMIZER_INDEX.items():
        v = embed_hparams(_valid_hparams(optimizer=name))
        assert v[5] == float(idx)


def test_embed_hparams_safe_for_zero_lr():
    v = embed_hparams(_valid_hparams(learning_rate=0.0))
    assert math.isfinite(v[0])  # clamped to 1e-12


def _write_log(entries: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _entry(**kwargs) -> dict:
    base = {
        "iteration": 1,
        "status": "accepted",
        "divergence_before": 2.0,
        "divergence_after": 1.0,
        "dim_before": 0.8,
        "dim_after": 1.3,
        "hparams_before": _valid_hparams(),
        "hparams_after": _valid_hparams(learning_rate=0.005),
        "patches": [{"file_path": "configs/hparams.yaml"}],
        "elapsed_s": 10.0,
        "summary": "dim 0.8 -> 1.3",
        "error": None,
        "logged_at": "2026-04-19T20:00:00",
    }
    base.update(kwargs)
    return base


def test_empty_log_produces_empty_trajectory():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "empty.jsonl"
        p.write_text("")
        mt = RepairHistoryReader().read(p)
        assert isinstance(mt, MetaTrajectory)
        assert mt.n_points == 0
        assert mt.n_transitions == 0


def test_nonexistent_log_handled_gracefully():
    mt = RepairHistoryReader().read("/tmp/does-not-exist-xyz.jsonl")
    assert mt.n_points == 0


def test_single_accepted_transition_adds_one_point():
    # trajectory starts with hparams_before, then the first accept adds one more point
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "r.jsonl"
        _write_log([_entry()], p)
        mt = RepairHistoryReader().read(p)
        assert mt.n_points == 2  # start + one accept
        assert mt.n_transitions == 1
        assert mt.states.shape == (2, 6)
        # lr dim went from log10(0.01)=-2 to log10(0.005)≈-2.301
        assert mt.states[0, 0] == pytest.approx(-2.0)
        assert mt.states[1, 0] == pytest.approx(math.log10(0.005))


def test_within_band_accept_does_not_add_point():
    # An "accepted" entry where hparams_after == hparams_before (e.g.
    # "already within band") is stationary — no new trajectory point.
    same = _valid_hparams()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "r.jsonl"
        _write_log([_entry(hparams_before=same, hparams_after=same,
                           summary="already within band")], p)
        mt = RepairHistoryReader().read(p)
        assert mt.n_points == 1


def test_no_fix_or_rollback_does_not_add_point():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "r.jsonl"
        _write_log([
            _entry(status="no_fix", hparams_after=None,
                   summary="no fix proposed"),
            _entry(status="no_improvement",
                   hparams_after=_valid_hparams(learning_rate=0.5)),
            _entry(status="rolled_back",
                   hparams_after=_valid_hparams(learning_rate=0.99)),
            _entry(status="error", hparams_after=None,
                   summary="LLM timeout"),
        ], p)
        mt = RepairHistoryReader().read(p)
        # Only the initial hparams_before of the first entry is kept.
        assert mt.n_points == 1


def test_multi_entry_accepts_compose_a_trajectory():
    h1 = _valid_hparams(learning_rate=0.1)
    h2 = _valid_hparams(learning_rate=0.01)
    h3 = _valid_hparams(learning_rate=0.001, optimizer="adamw")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "r.jsonl"
        _write_log([
            _entry(hparams_before=h1, hparams_after=h2,
                   divergence_before=3.0, divergence_after=2.0),
            _entry(hparams_before=h2, hparams_after=h3,
                   divergence_before=2.0, divergence_after=0.5),
        ], p)
        mt = RepairHistoryReader().read(p)
        assert mt.n_points == 3
        assert mt.n_transitions == 2
        assert mt.hparams_sequence[0]["learning_rate"] == 0.1
        assert mt.hparams_sequence[-1]["learning_rate"] == 0.001
        assert mt.divergence_scores == [3.0, 2.0, 0.5]


def test_multiple_log_files_concatenate():
    h_start = _valid_hparams(learning_rate=0.1)
    h_mid = _valid_hparams(learning_rate=0.01)
    h_end = _valid_hparams(learning_rate=0.001)
    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "run1.jsonl"
        p2 = Path(tmp) / "run2.jsonl"
        _write_log([_entry(hparams_before=h_start, hparams_after=h_mid)], p1)
        _write_log([_entry(hparams_before=h_mid, hparams_after=h_end)], p2)
        mt = RepairHistoryReader().read([p1, p2])
        assert mt.n_points == 3
        assert mt.sources == [p1, p2]


def test_malformed_json_lines_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "r.jsonl"
        content = "\n".join([
            "",
            "this is not json",
            json.dumps(_entry()),
            "{broken",
        ])
        p.write_text(content)
        mt = RepairHistoryReader().read(p)
        assert mt.n_points == 2  # only the valid entry contributed


def test_reads_real_sprint3_log():
    """If FractalTrainer/results/repair_log.jsonl exists, we should parse it."""
    repo_root = Path(__file__).resolve().parent.parent
    log_path = repo_root / "results" / "repair_log.jsonl"
    if not log_path.exists():
        pytest.skip("no real Sprint 3 log present")
    mt = RepairHistoryReader().read(log_path)
    assert mt.n_points >= 1
    assert mt.states.shape[1] == 6
