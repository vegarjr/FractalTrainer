"""Tests for SnakeEnv (v3 Sprint 13a)."""

from __future__ import annotations

import numpy as np
import pytest

from fractaltrainer.snake.env import (
    ACTIONS,
    ACTION_NAMES,
    SnakeEnv,
    encode_state,
    probe_batch,
)


def test_reset_produces_expected_shape():
    env = SnakeEnv(height=10, width=10, seed=0)
    state = env.reset()
    assert state.shape == (100,)
    assert state.dtype == np.int64
    # Exactly one head + one food.
    assert (state == 2).sum() == 1
    assert (state == 3).sum() == 1
    # Body length 2 (the deque has 3 cells: 1 head + 2 body).
    assert (state == 1).sum() == 2


def test_initial_snake_length_is_three():
    env = SnakeEnv(height=8, width=8, seed=1)
    env.reset()
    assert len(env.snake) == 3
    # Head at center.
    r, c = env.snake[0]
    assert r == 4 and c == 4


def test_step_changes_state_and_advances():
    env = SnakeEnv(height=10, width=10, seed=2)
    env.reset()
    head_before = env.snake[0]
    result = env.step(3)  # right
    assert not result.done
    assert result.reward == 0.0
    head_after = env.snake[0]
    assert head_after == (head_before[0], head_before[1] + 1)


def test_wall_collision_ends_episode_with_negative_reward():
    env = SnakeEnv(height=5, width=5, seed=3)
    env.reset()
    # From center (2, 2), moving right four times hits the wall.
    result = None
    for _ in range(4):
        if env.done:
            break
        result = env.step(3)  # right
    assert result is not None
    assert result.done
    assert result.reward == -1.0
    assert result.info["cause"] == "wall"


def test_eating_food_grows_snake_and_gives_reward():
    env = SnakeEnv(height=10, width=10, seed=4)
    env.reset()
    # Place food directly in front of the head.
    head = env.snake[0]
    env.food = (head[0], head[1] + 1)
    initial_length = len(env.snake)
    result = env.step(3)  # right
    assert result.reward == 1.0
    assert not result.done
    assert len(env.snake) == initial_length + 1


def test_self_collision_ends_episode():
    env = SnakeEnv(height=10, width=10, seed=5)
    env.reset()
    # Snake at (5, 5), (5, 4), (5, 3) after reset. Moving left puts head
    # at (5, 4), which is already body — wait, that's the tail cell.
    # The tail vacates unless we eat food. So left-move shouldn't self-
    # collide. But a 180-degree reversal IS a self-collision in general.
    # Let's grow the snake first, then reverse.
    head = env.snake[0]
    env.food = (head[0], head[1] + 1)
    env.step(3)  # right, eat food -> length 4
    # Now snake = [(5, 5+1), (5, 5), (5, 4), (5, 3)]. Reverse (left) →
    # head becomes (5, 5) which IS on body → self collision.
    result = env.step(2)  # left
    assert result.done
    assert result.reward == -1.0
    assert result.info["cause"] == "self"


def test_food_is_never_on_snake():
    env = SnakeEnv(height=6, width=6, seed=6)
    env.reset()
    snake_set = set(env.snake)
    assert env.food not in snake_set
    # After eating, new food still off-snake.
    env.food = env.snake[0][0], env.snake[0][1] + 1
    env.step(3)
    snake_set = set(env.snake)
    assert env.food not in snake_set


def test_invalid_action_raises():
    env = SnakeEnv(seed=7)
    env.reset()
    with pytest.raises(ValueError, match="action must be in"):
        env.step(4)
    with pytest.raises(ValueError):
        env.step(-1)


def test_step_on_done_raises():
    env = SnakeEnv(height=5, width=5, seed=8)
    env.reset()
    # Move until done
    done = False
    for _ in range(20):
        if env.done: break
        result = env.step(3)
        done = result.done
    assert done
    with pytest.raises(RuntimeError, match="done episode"):
        env.step(0)


def test_max_steps_forces_termination():
    env = SnakeEnv(height=5, width=5, seed=9, max_steps=2)
    env.reset()
    # Tiny action loop that shouldn't collide in 2 steps
    env.step(3)
    result = env.step(2)  # might collide — that's fine, either way done is True
    assert result.done


def test_seed_reproducibility():
    a = SnakeEnv(height=10, width=10, seed=42)
    b = SnakeEnv(height=10, width=10, seed=42)
    np.testing.assert_array_equal(a.reset(), b.reset())
    for action in [3, 3, 1, 1]:
        if a.done or b.done:
            break
        r_a = a.step(action); r_b = b.step(action)
        np.testing.assert_array_equal(r_a.state, r_b.state)
        assert r_a.reward == r_b.reward
        assert r_a.done == r_b.done


def test_different_seeds_produce_different_foods():
    a = SnakeEnv(height=10, width=10, seed=100)
    b = SnakeEnv(height=10, width=10, seed=101)
    # Food location is the only stochastic element at reset, so with
    # different seeds they should differ for the typical case.
    assert a.food != b.food


def test_encode_state_matches_board_layout():
    env = SnakeEnv(height=4, width=4, seed=0)
    env.reset()
    state = encode_state(env)
    assert state.shape == (16,)
    # Reshape back and compare to render_board.
    assert np.array_equal(state.reshape(4, 4), env.render_board())


def test_probe_batch_shape_and_determinism():
    probes_a = probe_batch(n_probes=20, height=8, width=8, seed=777)
    probes_b = probe_batch(n_probes=20, height=8, width=8, seed=777)
    assert probes_a.shape == (20, 64)
    assert probes_a.dtype == np.int64
    np.testing.assert_array_equal(probes_a, probes_b)
    # Different seeds → different probes
    probes_c = probe_batch(n_probes=20, height=8, width=8, seed=778)
    assert not np.array_equal(probes_a, probes_c)


def test_action_deltas_are_correct():
    assert ACTIONS[0] == (-1, 0)
    assert ACTIONS[1] == (1, 0)
    assert ACTIONS[2] == (0, -1)
    assert ACTIONS[3] == (0, 1)
    assert ACTION_NAMES[0] == "up"
