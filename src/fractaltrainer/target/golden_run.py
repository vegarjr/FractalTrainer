"""Golden-run trajectory matching.

Motivation (from Reviews/05_v2_sprint2_science_question.md): the v1 target
of "correlation dim ≈ 1.5" was shown empirically not to correlate with
test accuracy on MNIST — and the specific band [1.2, 1.8] is in fact
anti-correlated with good training in that setup. Picking a scalar dim
target a priori is bad epistemology.

The golden-run approach inverts this: first pick a *run that produced
good training* (high test accuracy), then record its trajectory's full
geometric signature. The repair loop's new job is to push the current
trajectory's signature toward the golden signature.

This sidesteps the "guess a target value" problem entirely. The target
shape is whatever shape the known-good trajectory actually has.

Signature features (9-d vector, pulled from existing trajectory metrics):
    correlation_dim
    total_path_length
    mean_step_norm
    step_norm_std
    dispersion
    displacement
    tortuosity
    mean_curvature
    recurrence_rate
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


SIGNATURE_FEATURES: tuple[str, ...] = (
    "correlation_dim",
    "total_path_length",
    "mean_step_norm",
    "step_norm_std",
    "dispersion",
    "displacement",
    "tortuosity",
    "mean_curvature",
    "recurrence_rate",
)


@dataclass
class GoldenRun:
    """Reference signature from a known-good training run.

    Attributes:
        name: identifying label (free-form).
        signature: dict mapping each SIGNATURE_FEATURE -> float value.
        test_accuracy: outcome metric the run achieved (if known). Used for
            provenance — we want to document *why* this run is "golden".
        hparams: the hparams that produced this run.
        source_trajectory: path to the .npy the signature was computed from
            (for reproducibility). Optional.
        notes: free-form notes, e.g. "top 1 of 48-run science sweep".
    """

    name: str
    signature: dict
    test_accuracy: float | None = None
    hparams: dict | None = None
    source_trajectory: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "GoldenRun":
        with open(path) as f:
            data = json.load(f)
        _validate_signature(data.get("signature", {}))
        return cls(
            name=data["name"],
            signature=data["signature"],
            test_accuracy=data.get("test_accuracy"),
            hparams=data.get("hparams"),
            source_trajectory=data.get("source_trajectory"),
            notes=data.get("notes"),
        )

    @classmethod
    def from_measurements(
        cls,
        name: str,
        correlation_dim_value: float,
        trajectory_metrics_dict: dict,
        test_accuracy: float | None = None,
        hparams: dict | None = None,
        source_trajectory: str | None = None,
        notes: str | None = None,
    ) -> "GoldenRun":
        sig = _build_signature(correlation_dim_value, trajectory_metrics_dict)
        return cls(
            name=name, signature=sig,
            test_accuracy=test_accuracy,
            hparams=hparams,
            source_trajectory=source_trajectory,
            notes=notes,
        )

    def as_vector(self) -> np.ndarray:
        """Return the 9-d feature vector in SIGNATURE_FEATURES order."""
        return _signature_to_vector(self.signature)

    def feature_scales(self, epsilon: float = 1e-6) -> np.ndarray:
        """Per-feature scale for z-normalization.

        Single-run version: use |value| + epsilon so each feature
        contributes on a comparable scale (roughly unit per feature).
        Ensemble golden runs could use actual per-feature std across
        runs (left as future work; would live in GoldenRunEnsemble).
        """
        v = self.as_vector()
        return np.abs(v) + epsilon


def _build_signature(correlation_dim_value: float,
                     trajectory_metrics_dict: dict) -> dict:
    merged = dict(trajectory_metrics_dict)
    merged["correlation_dim"] = correlation_dim_value
    for k in SIGNATURE_FEATURES:
        if k not in merged:
            merged[k] = float("nan")
    return {k: _coerce_finite(merged[k]) for k in SIGNATURE_FEATURES}


def _coerce_finite(x) -> float:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return xf


def _validate_signature(sig: dict) -> None:
    missing = [f for f in SIGNATURE_FEATURES if f not in sig]
    if missing:
        raise ValueError(
            f"golden-run signature missing features: {missing}")


def _signature_to_vector(sig: dict) -> np.ndarray:
    _validate_signature(sig)
    return np.array(
        [float(sig[f]) for f in SIGNATURE_FEATURES],
        dtype=np.float64,
    )


def build_signature_from_report(comparison_report: dict) -> dict:
    """Extract the 9-d signature from a run_comparison.py report dict."""
    primary = comparison_report.get("primary_result") or {}
    dim_value = primary.get("dim", float("nan"))
    if dim_value is None:
        dim_value = float("nan")
    traj = (comparison_report.get("baseline_metrics") or {}).get(
        "trajectory_metrics") or {}
    return _build_signature(float(dim_value), traj)


def golden_run_distance(
    current_signature: dict,
    golden: GoldenRun,
    ignore_nan_features: bool = True,
    epsilon: float = 1e-6,
) -> float:
    """z-normalized Euclidean distance between a current trajectory's
    signature and a golden run's signature.

    Distance == 0 means the signatures are identical. Distance ≈ 1 means
    the current signature is off by one unit-scale per feature on
    average. Infinity (or very large) means catastrophic shape mismatch.

    NaNs: by default NaN features are dropped from both sides before
    distance is computed (useful because a very short trajectory can
    produce NaN in correlation_dim without meaning the other features
    are also invalid). Setting ignore_nan_features=False treats any NaN
    as an infinite distance contribution.
    """
    golden_vec = golden.as_vector()
    current_vec = _signature_to_vector(current_signature)
    scales = np.abs(golden_vec) + epsilon

    diff = current_vec - golden_vec
    z = diff / scales

    if ignore_nan_features:
        mask = np.isfinite(current_vec) & np.isfinite(golden_vec)
        if not mask.any():
            return float("inf")
        z = z[mask]
    else:
        if not np.all(np.isfinite(current_vec)) or not np.all(
                np.isfinite(golden_vec)):
            return float("inf")

    return float(np.linalg.norm(z))


def golden_run_per_feature_deltas(
    current_signature: dict,
    golden: GoldenRun,
    epsilon: float = 1e-6,
) -> dict:
    """Return per-feature z-score differences as a dict (useful for prompts)."""
    golden_vec = golden.as_vector()
    current_vec = _signature_to_vector(current_signature)
    scales = np.abs(golden_vec) + epsilon
    z = (current_vec - golden_vec) / scales
    result: dict = {}
    for i, feat in enumerate(SIGNATURE_FEATURES):
        cv = float(current_vec[i])
        gv = float(golden_vec[i])
        result[feat] = {
            "current": cv if math.isfinite(cv) else None,
            "golden": gv if math.isfinite(gv) else None,
            "z_delta": float(z[i]) if math.isfinite(z[i]) else None,
        }
    return result
