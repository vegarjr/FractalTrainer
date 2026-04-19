from fractaltrainer.target.target_shape import TargetShape, ProjectionSpec, load_target
from fractaltrainer.target.divergence import divergence_score, should_intervene, within_band
from fractaltrainer.target.golden_run import (
    GoldenRun,
    SIGNATURE_FEATURES,
    build_signature_from_report,
    golden_run_distance,
    golden_run_per_feature_deltas,
)

__all__ = [
    "TargetShape",
    "ProjectionSpec",
    "load_target",
    "divergence_score",
    "should_intervene",
    "within_band",
    "GoldenRun",
    "SIGNATURE_FEATURES",
    "build_signature_from_report",
    "golden_run_distance",
    "golden_run_per_feature_deltas",
]
