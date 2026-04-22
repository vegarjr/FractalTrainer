"""v3 Sprint 17 — closed perception→growth loop + context injection."""

from fractaltrainer.integration.context_mlp import (
    ContextAwareCNN,
    ContextAwareMLP,
    baseline_mlp_forward,
)
from fractaltrainer.integration.context_injection import (
    ContextSpec,
    gather_context,
    random_context,
)
from fractaltrainer.integration.describer_adapter import (
    DescribeResult,
    Describer,
    MockDescriber,
    OracleDescriber,
)
from fractaltrainer.integration.recluster import (
    ClusteringResult,
    recluster,
    agglomerative_cluster_average_linkage,
)
from fractaltrainer.integration.self_spawn import (
    AutoSpawnPolicy,
    SpawnProposal,
)
from fractaltrainer.integration.signatures import (
    SignatureFn,
    get_signature_fn,
    penultimate_signature,
    softmax_signature,
)
from fractaltrainer.integration.spawn import (
    TrainStats,
    spawn_baseline,
    spawn_random_context,
    spawn_with_context,
)
from fractaltrainer.integration.pipeline import (
    FractalPipeline,
    PipelineStep,
    QueryInput,
    ReclusterPolicy,
)
from fractaltrainer.integration.evaluation import (
    BudgetResult,
    SampleEfficiencyResult,
    evaluate_expert,
    render_efficiency_table_md,
    sample_efficiency_curve,
)

__all__ = [
    "ContextAwareCNN", "ContextAwareMLP", "baseline_mlp_forward",
    "ContextSpec", "gather_context", "random_context",
    "DescribeResult", "Describer", "MockDescriber", "OracleDescriber",
    "ClusteringResult", "recluster", "agglomerative_cluster_average_linkage",
    "AutoSpawnPolicy", "SpawnProposal",
    "SignatureFn", "get_signature_fn", "penultimate_signature", "softmax_signature",
    "TrainStats", "spawn_baseline", "spawn_random_context", "spawn_with_context",
    "FractalPipeline", "PipelineStep", "QueryInput", "ReclusterPolicy",
    "BudgetResult", "SampleEfficiencyResult",
    "evaluate_expert", "render_efficiency_table_md", "sample_efficiency_curve",
]
