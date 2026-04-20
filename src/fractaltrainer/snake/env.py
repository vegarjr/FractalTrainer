"""SnakeEnv — deterministic grid-world Snake simulator.

Minimal, deterministic, seeded. No pygame, no rendering — this is pure
logic so it runs fast in training and testing loops.

Game rules:
  * Board is H × W (default 10 × 10), toroidal = False (walls kill).
  * Snake is a deque of (row, col) tuples, head at index 0.
  * Food is a single (row, col) tuple, not on the snake.
  * Each step: move head by the chosen action direction; if the new
    head is on food, grow; else pop tail. Collision (wall or self) →
    episode ends with reward = -1. Food eaten → reward = +1. Otherwise
    reward = 0 per move.
  * Max steps per episode = H * W * 3 (prevents infinite loops in
    circle-walking policies).

State encoding for behavior cloning:
    A flat (H * W,) numpy int array with channel values:
        0 = empty
        1 = snake body (non-head)
        2 = snake head
        3 = food
    Row-major order. MLPs take this as-is; no one-hot expansion.

Probe-batch methodology (registry signatures):
    A canonical set of N game states (random board configurations with
    a fixed seed) that every trained policy is evaluated on to produce
    a 4-way action distribution per state. The flattened (N, 4) matrix
    becomes the policy's signature in FractalRegistry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np


# (drow, dcol) deltas for each action index
ACTIONS = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
}
ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}


@dataclass
class StepResult:
    """Return value of SnakeEnv.step."""
    state: np.ndarray
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class SnakeEnv:
    """A deterministic, seedable Snake simulator on an H×W grid.

    Usage:
        env = SnakeEnv(height=10, width=10, seed=42)
        state = env.reset()
        while not done:
            action = policy(state, env.snake, env.food)  # 0..3
            result = env.step(action)
            state = result.state
            done = result.done

    Public attributes:
        height, width : grid dimensions.
        snake         : current deque of (row, col) with head at index 0.
        food          : current (row, col) food position.
        steps         : steps taken this episode.
        score         : food eaten this episode.
    """

    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        self.height = height
        self.width = width
        self.max_steps = max_steps if max_steps is not None else 3 * height * width
        self._rng = np.random.RandomState(seed)
        self.snake: Deque[tuple[int, int]] = deque()
        self.food: tuple[int, int] = (0, 0)
        self.steps: int = 0
        self.score: int = 0
        self._done: bool = True
        self.reset()

    # ── State access ──────────────────────────────────────────────

    def render_board(self) -> np.ndarray:
        """Return the board as an (H, W) numpy int array using the
        encoding scheme {0 empty, 1 body, 2 head, 3 food}."""
        board = np.zeros((self.height, self.width), dtype=np.int64)
        if self.food is not None:
            board[self.food[0], self.food[1]] = 3
        for i, (r, c) in enumerate(self.snake):
            board[r, c] = 2 if i == 0 else 1
        return board

    def get_state(self) -> np.ndarray:
        """Flattened state vector suitable for MLP input."""
        return self.render_board().flatten()

    # ── Dynamics ──────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Re-initialize the episode. Snake at center (length 3),
        food at a random empty cell."""
        self.snake = deque()
        mid_r, mid_c = self.height // 2, self.width // 2
        self.snake.append((mid_r, mid_c))
        self.snake.append((mid_r, mid_c - 1))
        self.snake.append((mid_r, mid_c - 2))
        self.food = self._spawn_food()
        self.steps = 0
        self.score = 0
        self._done = False
        return self.get_state()

    def _spawn_food(self) -> tuple[int, int]:
        body = set(self.snake)
        free = [(r, c) for r in range(self.height)
                for c in range(self.width)
                if (r, c) not in body]
        if not free:
            # Board full: arbitrary cell (episode probably ends next anyway).
            return (0, 0)
        idx = int(self._rng.randint(0, len(free)))
        return free[idx]

    def step(self, action: int) -> StepResult:
        """Advance one step. Returns a StepResult."""
        if self._done:
            raise RuntimeError("step() on a done episode — call reset()")
        if action not in ACTIONS:
            raise ValueError(f"action must be in {list(ACTIONS)}; got {action!r}")

        dr, dc = ACTIONS[action]
        head = self.snake[0]
        new_head = (head[0] + dr, head[1] + dc)
        self.steps += 1

        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.height
                or new_head[1] < 0 or new_head[1] >= self.width):
            self._done = True
            return StepResult(
                state=self.get_state(), reward=-1.0, done=True,
                info={"cause": "wall"},
            )

        # Self-collision — check against body EXCLUDING the tail cell,
        # because the tail will vacate as part of this step (unless we
        # just ate food).
        body_for_collision = list(self.snake)
        ate = (new_head == self.food)
        if not ate:
            body_for_collision = body_for_collision[:-1]
        if new_head in body_for_collision:
            self._done = True
            return StepResult(
                state=self.get_state(), reward=-1.0, done=True,
                info={"cause": "self"},
            )

        # Valid move
        self.snake.appendleft(new_head)
        reward = 0.0
        if ate:
            self.score += 1
            reward = 1.0
            self.food = self._spawn_food()
        else:
            self.snake.pop()

        if self.steps >= self.max_steps:
            self._done = True
            return StepResult(
                state=self.get_state(), reward=reward, done=True,
                info={"cause": "max_steps", "score": self.score},
            )

        return StepResult(state=self.get_state(), reward=reward,
                          done=False, info={"score": self.score})

    # ── Convenience ───────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._done


# ── Probe-batch (for registry signatures) ─────────────────────────

def encode_state(env: SnakeEnv) -> np.ndarray:
    """Alias for env.get_state() to emphasize the encoding contract."""
    return env.get_state()


def probe_batch(n_probes: int = 50, height: int = 10, width: int = 10,
                seed: int = 12345) -> np.ndarray:
    """Produce a canonical (n_probes, height*width) batch of game states
    for computing policy signatures.

    Each probe is a random snapshot of a Snake game (not a full
    trajectory — just the board state at some random point). Different
    policies see the same probes in the same order.

    Reproducible per seed.
    """
    rng = np.random.RandomState(seed)
    probes = np.zeros((n_probes, height * width), dtype=np.int64)
    for i in range(n_probes):
        env = SnakeEnv(height=height, width=width,
                        seed=int(rng.randint(0, 2**31 - 1)))
        # Play a few random valid moves to diversify the state.
        n_random_moves = int(rng.randint(0, 10))
        for _ in range(n_random_moves):
            if env.done:
                break
            action = int(rng.randint(0, 4))
            env.step(action)
        probes[i] = env.get_state()
    return probes
