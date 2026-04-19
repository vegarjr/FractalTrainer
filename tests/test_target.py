"""Tests for target_shape and divergence."""

import math
from pathlib import Path

import pytest

from fractaltrainer.target.target_shape import TargetShape, ProjectionSpec, load_target
from fractaltrainer.target.divergence import divergence_score, should_intervene, within_band


REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_target() -> TargetShape:
    return TargetShape(
        dim_target=1.5,
        tolerance=0.3,
        hysteresis=1.2,
        projection=ProjectionSpec(),
    )


def test_load_default_target_yaml():
    t = load_target(REPO_ROOT / "configs" / "target_shape.yaml")
    assert t.dim_target == 1.5
    assert t.tolerance == 0.3
    assert t.method == "correlation_dim"
    assert t.projection.method == "random_proj"
    assert t.projection.n_components == 16


def test_target_band_properties():
    t = _default_target()
    assert t.band_low == pytest.approx(1.2)
    assert t.band_high == pytest.approx(1.8)


def test_divergence_score_at_center_is_zero():
    t = _default_target()
    assert divergence_score(1.5, t) == pytest.approx(0.0)


def test_divergence_score_at_band_edge_is_one():
    t = _default_target()
    assert divergence_score(1.2, t) == pytest.approx(1.0)
    assert divergence_score(1.8, t) == pytest.approx(1.0)


def test_divergence_score_outside_band_exceeds_one():
    t = _default_target()
    assert divergence_score(2.3, t) > 1.0
    assert divergence_score(0.0, t) > 1.0


def test_divergence_score_nan_is_infinite():
    t = _default_target()
    assert math.isinf(divergence_score(float("nan"), t))


def test_should_intervene_hysteresis():
    t = _default_target()
    # Inside hysteresis (score=1.1 < hysteresis=1.2) → do not intervene
    assert not should_intervene(1.1, t)
    # Outside hysteresis (score=1.3 > 1.2) → intervene
    assert should_intervene(1.3, t)
    # NaN/inf → intervene
    assert should_intervene(float("nan"), t)
    assert should_intervene(float("inf"), t)


def test_within_band_matches_band_bounds():
    t = _default_target()
    assert within_band(1.5, t)
    assert within_band(1.2, t)
    assert within_band(1.8, t)
    assert not within_band(1.19, t)
    assert not within_band(1.81, t)
    assert not within_band(float("nan"), t)


def test_load_invalid_target_rejects():
    import tempfile
    import yaml as _yaml

    # tolerance <= 0
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        _yaml.dump({"dim_target": 1.5, "tolerance": 0.0}, f)
        path = f.name
    with pytest.raises(ValueError):
        load_target(path)

    # unknown method
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        _yaml.dump({"dim_target": 1.5, "tolerance": 0.3, "method": "bogus"}, f)
        path = f.name
    with pytest.raises(ValueError):
        load_target(path)
