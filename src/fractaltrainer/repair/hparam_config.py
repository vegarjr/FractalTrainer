"""Hyperparameter schema, YAML IO, and validation.

Only the keys listed in HPARAM_SCHEMA may appear in a patched hparams.yaml.
Values must pass the type + range/allowlist check in validate_hparams().
Anything else is rejected before application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# key -> (type, low, high) for numeric; (type, set) for categorical
HPARAM_SCHEMA: dict[str, tuple] = {
    "learning_rate": (float, 1e-6, 1.0),
    "batch_size":    (int,   1,    1024),
    "weight_decay":  (float, 0.0,  0.1),
    "dropout":       (float, 0.0,  0.9),
    "init_seed":     (int,   0,    2**31 - 1),
    "optimizer":     (str,   {"sgd", "adam", "adamw"}),
}


def load_hparams(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected YAML mapping at top level")
    return data


def save_hparams(hparams: dict, path: str | Path) -> None:
    errs = validate_hparams(hparams)
    if errs:
        raise ValueError(f"refusing to save invalid hparams: {errs}")
    with open(path, "w") as f:
        yaml.safe_dump(hparams, f, sort_keys=False)


def validate_hparams(hparams: dict) -> list[str]:
    """Return a list of human-readable errors. Empty list means valid."""
    errors: list[str] = []

    if not isinstance(hparams, dict):
        return [f"hparams must be a dict, got {type(hparams).__name__}"]

    unknown = set(hparams.keys()) - set(HPARAM_SCHEMA.keys())
    for k in unknown:
        errors.append(f"unknown hparam key: {k!r}")

    required = set(HPARAM_SCHEMA.keys())
    missing = required - set(hparams.keys())
    for k in missing:
        errors.append(f"missing required hparam: {k!r}")

    for key, value in hparams.items():
        if key not in HPARAM_SCHEMA:
            continue
        spec = HPARAM_SCHEMA[key]
        kind = spec[0]
        if kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"{key}: expected int, got {type(value).__name__}")
                continue
            low, high = spec[1], spec[2]
            if not (low <= value <= high):
                errors.append(f"{key}: {value} outside [{low}, {high}]")
        elif kind is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{key}: expected float, got {type(value).__name__}")
                continue
            low, high = spec[1], spec[2]
            v = float(value)
            if not (low <= v <= high):
                errors.append(f"{key}: {v} outside [{low}, {high}]")
        elif kind is str:
            if not isinstance(value, str):
                errors.append(f"{key}: expected str, got {type(value).__name__}")
                continue
            allowed = spec[1]
            if value not in allowed:
                errors.append(f"{key}: {value!r} not in {sorted(allowed)}")

    return errors
