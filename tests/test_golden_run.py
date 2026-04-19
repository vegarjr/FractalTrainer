"""Tests for golden-run trajectory matching (v2 Sprint 3 candidate A)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from fractaltrainer.target.golden_run import (
    GoldenRun,
    SIGNATURE_FEATURES,
    build_signature_from_report,
    golden_run_distance,
    golden_run_per_feature_deltas,
)
from fractaltrainer.target.target_shape import (
    ProjectionSpec,
    TargetShape,
    load_target,
)
from fractaltrainer.target.divergence import (
    divergence_score,
    should_intervene,
    within_band,
)


def _sig(dim=0.7, path=10.0, mean_step=0.5, std=0.1, disp=1.2, displace=5.0,
         tort=2.0, curv=1.5, recur=0.1) -> dict:
    return {
        "correlation_dim": dim,
        "total_path_length": path,
        "mean_step_norm": mean_step,
        "step_norm_std": std,
        "dispersion": disp,
        "displacement": displace,
        "tortuosity": tort,
        "mean_curvature": curv,
        "recurrence_rate": recur,
    }


def _golden(**overrides) -> GoldenRun:
    return GoldenRun(name="test", signature=_sig(**overrides),
                     test_accuracy=0.9)


# ── GoldenRun basics ────────────────────────────────────────────────

def test_signature_features_has_exactly_nine_entries():
    assert len(SIGNATURE_FEATURES) == 9


def test_as_vector_matches_feature_order():
    g = _golden(dim=1.0, path=2.0, mean_step=3.0)
    v = g.as_vector()
    assert v.shape == (9,)
    # First three must be dim, path_length, mean_step_norm in that order
    assert v[0] == 1.0
    assert v[1] == 2.0
    assert v[2] == 3.0


def test_save_load_roundtrip():
    g = _golden()
    g.test_accuracy = 0.92
    g.hparams = {"learning_rate": 0.1, "optimizer": "sgd"}
    g.notes = "golden run from sweep"
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "g.json"
        g.save(p)
        loaded = GoldenRun.load(p)
        assert loaded.signature == g.signature
        assert loaded.test_accuracy == 0.92
        assert loaded.hparams == g.hparams
        assert loaded.notes == "golden run from sweep"


def test_load_rejects_missing_features():
    bad = {"name": "bad", "signature": {"correlation_dim": 1.0}}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="missing features"):
            GoldenRun.load(p)


def test_from_measurements_builds_signature():
    g = GoldenRun.from_measurements(
        name="run38",
        correlation_dim_value=0.744,
        trajectory_metrics_dict={
            "total_path_length": 13.0,
            "mean_step_norm": 0.52,
            "step_norm_std": 0.22,
            "dispersion": 1.41,
            "displacement": 5.55,
            "tortuosity": 2.34,
            "mean_curvature": 1.54,
            "recurrence_rate": 0.0,
        },
        test_accuracy=0.944,
        hparams={"learning_rate": 0.1, "optimizer": "sgd", "init_seed": 101},
    )
    assert g.signature["correlation_dim"] == pytest.approx(0.744)
    assert g.signature["tortuosity"] == pytest.approx(2.34)
    assert g.test_accuracy == 0.944


# ── Distance ────────────────────────────────────────────────────────

def test_match_self_is_zero():
    g = _golden()
    d = golden_run_distance(g.signature, g)
    assert d == pytest.approx(0.0)


def test_completely_different_signature_has_large_distance():
    g = _golden(dim=0.7, tort=2.0)
    # Wildly different signature
    far = _sig(dim=3.0, tort=10.0, path=100.0)
    d = golden_run_distance(far, g)
    assert d > 3.0


def test_small_perturbation_has_small_distance():
    g = _golden(dim=0.7, mean_step=0.5, tort=2.0, path=10.0, disp=1.2,
                displace=5.0, std=0.1, curv=1.5, recur=0.1)
    # Perturb each feature by 10%
    s = dict(g.signature)
    for k in list(s.keys()):
        s[k] = s[k] * 1.1 if s[k] != 0 else 0.01
    d = golden_run_distance(s, g)
    # z-normalized → each feature contributes ~0.1 → L2 ~ sqrt(9 * 0.01) ~ 0.3
    assert 0.2 < d < 0.7


def test_per_feature_deltas_shape():
    g = _golden()
    d = golden_run_per_feature_deltas(g.signature, g)
    assert set(d.keys()) == set(SIGNATURE_FEATURES)
    for feat in SIGNATURE_FEATURES:
        assert "current" in d[feat]
        assert "golden" in d[feat]
        assert "z_delta" in d[feat]
        # self-match → z_delta = 0
        assert d[feat]["z_delta"] == pytest.approx(0.0)


def test_nan_features_handled_gracefully():
    g = _golden(dim=0.7)
    s = _sig(dim=float("nan"))  # correlation_dim is NaN
    d = golden_run_distance(s, g, ignore_nan_features=True)
    # Other 8 features match exactly → distance should be finite + small
    assert d == pytest.approx(0.0)
    d2 = golden_run_distance(s, g, ignore_nan_features=False)
    assert d2 == float("inf")


# ── build_signature_from_report ─────────────────────────────────────

def test_build_signature_from_comparison_report():
    report = {
        "primary_result": {"method": "correlation_dim", "dim": 0.75},
        "baseline_metrics": {
            "trajectory_metrics": _sig(dim=999.0),  # dim here is ignored
        },
    }
    sig = build_signature_from_report(report)
    # correlation_dim should come from primary_result, NOT trajectory_metrics
    assert sig["correlation_dim"] == pytest.approx(0.75)
    assert sig["tortuosity"] == pytest.approx(2.0)


# ── TargetShape extension ──────────────────────────────────────────

def test_target_shape_loads_golden_run_path():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.yaml"
        p.write_text(
            "dim_target: 1.0\n"
            "tolerance: 0.5\n"
            "method: golden_run_match\n"
            "golden_run_path: golden_runs/good.json\n"
            "projection:\n  method: random_proj\n  n_components: 16\n  seed: 0\n"
        )
        t = load_target(p)
        assert t.method == "golden_run_match"
        assert t.golden_run_path == "golden_runs/good.json"


def test_target_shape_rejects_golden_match_without_path():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.yaml"
        p.write_text(
            "dim_target: 1.0\ntolerance: 0.5\nmethod: golden_run_match\n"
        )
        with pytest.raises(ValueError, match="golden_run_path"):
            load_target(p)


def test_target_shape_default_method_unchanged():
    # Backward-compat: loading a v1 target_shape.yaml with no method
    # still works (method defaults to correlation_dim).
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.yaml"
        p.write_text(
            "dim_target: 1.5\ntolerance: 0.3\n"
            "projection:\n  method: random_proj\n  n_components: 16\n  seed: 0\n"
        )
        t = load_target(p)
        assert t.method == "correlation_dim"
        assert t.golden_run_path is None


# ── Unified divergence_score ────────────────────────────────────────

def test_divergence_score_scalar_method_unchanged():
    t = TargetShape(dim_target=1.5, tolerance=0.3,
                    method="correlation_dim",
                    projection=ProjectionSpec())
    # Scalar passthrough still works
    assert divergence_score(1.5, t) == pytest.approx(0.0)
    assert divergence_score(1.8, t) == pytest.approx(1.0)
    assert divergence_score(2.1, t) == pytest.approx(2.0)
    # Dict form should also work and pull correlation_dim
    assert divergence_score({"correlation_dim": 1.5}, t) == pytest.approx(0.0)


def test_divergence_score_golden_run_match():
    g = _golden()
    with tempfile.TemporaryDirectory() as tmp:
        gp = Path(tmp) / "golden.json"
        g.save(gp)
        t = TargetShape(dim_target=1.0, tolerance=0.5,
                        method="golden_run_match",
                        projection=ProjectionSpec(),
                        golden_run_path=str(gp))
        # Self-match → 0
        assert divergence_score(g.signature, t) == pytest.approx(0.0)
        # Different sig → positive
        far = _sig(dim=3.0, tort=10.0, path=100.0)
        assert divergence_score(far, t) > 3.0


def test_within_band_golden_method():
    g = _golden()
    with tempfile.TemporaryDirectory() as tmp:
        gp = Path(tmp) / "golden.json"
        g.save(gp)
        t = TargetShape(dim_target=1.0, tolerance=0.5,
                        method="golden_run_match",
                        projection=ProjectionSpec(),
                        golden_run_path=str(gp))
        assert within_band(g.signature, t) is True
        far = _sig(dim=3.0, tort=10.0, path=100.0)
        assert within_band(far, t) is False


def test_should_intervene_unchanged_for_golden():
    g = _golden()
    with tempfile.TemporaryDirectory() as tmp:
        gp = Path(tmp) / "golden.json"
        g.save(gp)
        t = TargetShape(dim_target=1.0, tolerance=0.5,
                        method="golden_run_match", hysteresis=1.2,
                        projection=ProjectionSpec(),
                        golden_run_path=str(gp))
        # self-match score 0 → do not intervene
        assert should_intervene(divergence_score(g.signature, t), t) is False
        # large divergence → intervene
        far = _sig(dim=3.0, tort=10.0)
        assert should_intervene(divergence_score(far, t), t) is True
