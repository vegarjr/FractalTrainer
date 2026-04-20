"""Snake sub-package (v3 Sprint 13).

Extends FractalTrainer's registry + routing machinery to a non-MNIST
task: a deterministic Snake grid-world where the "experts" are policies
trained by behavior-cloning from LLM-generated strategies.

Modules:
    env             : SnakeEnv simulator + state encoding
    teacher         : prompt the LLM for strategies, collect demonstrations
    behavior_clone  : train a policy MLP from demonstrations
"""

from fractaltrainer.snake.env import (
    ACTIONS,
    SnakeEnv,
    encode_state,
    probe_batch,
)
from fractaltrainer.snake.teacher import (
    STRATEGY_PROMPTS,
    Demo,
    collect_demos,
    generate_demos_from_llm,
)
from fractaltrainer.snake.behavior_clone import (
    PolicyMLP,
    TrainedPolicy,
    evaluate_ensemble,
    evaluate_single_policy,
    play_games,
    train_policy,
)

__all__ = [
    "ACTIONS",
    "SnakeEnv",
    "encode_state",
    "probe_batch",
    "STRATEGY_PROMPTS",
    "Demo",
    "collect_demos",
    "generate_demos_from_llm",
    "PolicyMLP",
    "TrainedPolicy",
    "evaluate_ensemble",
    "evaluate_single_policy",
    "play_games",
    "train_policy",
]
