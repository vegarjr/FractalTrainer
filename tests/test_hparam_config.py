"""Tests for hparam_config — schema enforcement."""

import tempfile
from pathlib import Path

import pytest

from fractaltrainer.repair.hparam_config import (
    HPARAM_SCHEMA,
    load_hparams,
    save_hparams,
    validate_hparams,
)


def _valid():
    return {
        "learning_rate": 0.01,
        "batch_size": 64,
        "weight_decay": 0.0001,
        "dropout": 0.1,
        "init_seed": 42,
        "optimizer": "adam",
    }


def test_schema_has_expected_keys():
    assert set(HPARAM_SCHEMA.keys()) == {
        "learning_rate", "batch_size", "weight_decay", "dropout",
        "init_seed", "optimizer",
    }


def test_valid_hparams_pass():
    assert validate_hparams(_valid()) == []


def test_unknown_key_rejected():
    h = _valid()
    h["mystery_knob"] = 3.14
    errs = validate_hparams(h)
    assert any("mystery_knob" in e for e in errs)


def test_missing_key_rejected():
    h = _valid()
    del h["learning_rate"]
    errs = validate_hparams(h)
    assert any("learning_rate" in e and "missing" in e for e in errs)


def test_out_of_range_rejected():
    h = _valid()
    h["learning_rate"] = 10.0
    errs = validate_hparams(h)
    assert any("learning_rate" in e and "outside" in e for e in errs)


def test_bad_type_rejected():
    h = _valid()
    h["batch_size"] = 3.14
    errs = validate_hparams(h)
    assert any("batch_size" in e for e in errs)


def test_bad_optimizer_rejected():
    h = _valid()
    h["optimizer"] = "momentum"
    errs = validate_hparams(h)
    assert any("optimizer" in e for e in errs)


def test_non_dict_rejected():
    errs = validate_hparams([1, 2, 3])  # type: ignore[arg-type]
    assert errs


def test_save_invalid_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bad.yaml"
        with pytest.raises(ValueError):
            save_hparams({"learning_rate": 100.0}, out)


def test_save_then_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "good.yaml"
        save_hparams(_valid(), out)
        loaded = load_hparams(out)
        assert loaded == _valid()
