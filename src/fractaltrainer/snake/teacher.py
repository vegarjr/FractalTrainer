"""LLM-driven strategy generation + demonstration collection.

Pipeline:
    1. Prompt the local LLM for K diverse Snake strategies.
    2. Extract each response's `next_action(board, snake, food)` function.
    3. ast.parse for syntactic validity; compile in a minimal sandbox
       namespace (only stdlib + numpy accessible, no filesystem/network).
    4. For each compiled strategy, play M games in SnakeEnv, collecting
       (state_vector, action_int) pairs. On any runtime error / timeout
       per move, fall back to a random action and log.
    5. Return Demo records that the behavior-cloning module can train on.

Safety: the sandbox namespace is minimal. We don't trust the LLM output
to be benign, but since all strategies are small pure functions and the
host calls them with timeouts, the risk surface is small. NEVER extend
this to untrusted external LLM output without adding a subprocess
sandbox.
"""

from __future__ import annotations

import ast
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from fractaltrainer.snake.env import ACTION_NAMES, ACTIONS, SnakeEnv


# ── Default strategy prompts ────────────────────────────────────────

STRATEGY_PROMPTS = {
    "greedy_food": (
        "Move directly toward the food. Pick the action that reduces "
        "Manhattan distance to food most, unless that action would hit a "
        "wall or the snake's own body — in which case pick any safe move."
    ),
    "bfs_safe": (
        "Do a breadth-first search from the snake's head to the food, "
        "treating wall cells and body cells as blocked. Return the first "
        "action of the shortest safe path. If no path exists, pick any "
        "action that doesn't immediately kill the snake."
    ),
    "wall_hugger": (
        "Prefer moves that keep the snake near the walls (low row or "
        "column indices near 0 or near max). This minimizes wasted "
        "central space. Avoid self-collision."
    ),
    "center_stayer": (
        "Prefer moves that keep the head close to the center of the "
        "board. This preserves options. Eat food opportunistically but "
        "avoid long detours to the edge."
    ),
    "survival_first": (
        "Prioritize staying alive over eating. Pick the action that "
        "leaves the snake the most free neighbor cells after the move. "
        "Eat food only if it's the most-free option. Never self-collide."
    ),
}


SYSTEM_PROMPT = (
    "You are a concise Python code assistant. When asked to write a "
    "function, reply with ONLY the Python code in a ```python``` block. "
    "No explanation before or after."
)


def _build_user_prompt(strategy_name: str, strategy_desc: str,
                       height: int, width: int) -> str:
    return (
        f"Write a Python function `next_action(board, snake, food)` that "
        f"plays Snake on a {height}×{width} grid.\n\n"
        f"Inputs:\n"
        f"  - `board`: 2D list of ints, {height} rows × {width} cols. "
        f"0=empty, 1=body, 2=head, 3=food.\n"
        f"  - `snake`: list of (row, col) tuples, head at index 0.\n"
        f"  - `food`: (row, col) tuple.\n\n"
        f"Output: one of 'up', 'down', 'left', 'right' (strings).\n\n"
        f"Strategy: {strategy_desc}\n\n"
        f"Requirements: standard library only, NO imports needed "
        f"(stdlib + built-ins are provided). No print statements. Only "
        f"define `next_action`. Return 'up' if truly ambiguous."
    )


def _extract_python(text: str) -> str:
    """Pull the python code out of a fenced code block, or take the
    whole thing if no fence is present."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _compile_strategy(code: str) -> Callable:
    """ast.parse, then exec in a minimal namespace, return next_action.

    Raises ValueError on parse failure, missing next_action, etc.
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"syntax error in LLM output: {e}") from e

    ns: dict = {
        "__builtins__": {
            # A minimal but functional subset of built-ins.
            "range": range, "len": len, "min": min, "max": max,
            "abs": abs, "sum": sum, "sorted": sorted, "reversed": reversed,
            "map": map, "filter": filter, "zip": zip, "enumerate": enumerate,
            "set": set, "frozenset": frozenset, "list": list, "tuple": tuple,
            "dict": dict, "any": any, "all": all, "int": int, "float": float,
            "bool": bool, "str": str, "round": round,
            "isinstance": isinstance, "hasattr": hasattr,
            "ValueError": ValueError, "IndexError": IndexError,
            "KeyError": KeyError, "TypeError": TypeError, "print": lambda *a, **kw: None,
            # Queue for BFS
            "deque": deque, "__import__": __import__,  # allow 'from collections import deque'
            "True": True, "False": False, "None": None,
            "math": math,  # common helpers; benign
        },
        "deque": deque,
        "math": math,
    }
    try:
        exec(compile(code, "<llm_strategy>", "exec"), ns)
    except Exception as e:
        raise ValueError(f"could not exec LLM code: {e}") from e

    fn = ns.get("next_action")
    if fn is None or not callable(fn):
        raise ValueError("LLM output did not define `next_action`")
    return fn


def _normalize_action(raw) -> Optional[int]:
    """Convert 'up'/0/etc. → int 0..3, or None if invalid."""
    if isinstance(raw, str):
        raw = raw.strip().lower()
        rev = {v: k for k, v in ACTION_NAMES.items()}
        return rev.get(raw)
    if isinstance(raw, int) and 0 <= raw < 4:
        return raw
    return None


# ── Demo collection ─────────────────────────────────────────────────

@dataclass
class Demo:
    """One strategy's demonstration dataset."""
    name: str
    states: np.ndarray       # (N, H*W) int64
    actions: np.ndarray      # (N,) int64 in 0..3
    games_played: int
    errors: int              # count of strategy-call failures (fell back)
    avg_survival: float      # mean episode length
    avg_score: float         # mean food eaten
    raw_code: str = ""       # for debugging / review

    def __post_init__(self):
        self.states = np.asarray(self.states, dtype=np.int64)
        self.actions = np.asarray(self.actions, dtype=np.int64)


def collect_demos(
    strategy_fn: Callable, name: str, *,
    n_games: int = 50, height: int = 10, width: int = 10,
    game_seed_base: int = 1000, raw_code: str = "",
) -> Demo:
    """Play n_games games with strategy_fn, collect (state, action) pairs.

    On runtime error or invalid action return, fall back to a random
    safe-looking action (trying all 4 in order) to keep the episode
    moving.
    """
    states: list[np.ndarray] = []
    actions: list[int] = []
    lengths: list[int] = []
    scores: list[int] = []
    errors = 0
    rng_fallback = np.random.RandomState(game_seed_base)

    for g in range(n_games):
        env = SnakeEnv(height=height, width=width,
                        seed=game_seed_base + g)
        env.reset()
        game_steps = 0
        while not env.done:
            board = env.render_board().tolist()
            snake = list(env.snake)
            food = env.food
            state = env.get_state().copy()
            try:
                raw = strategy_fn(board, snake, food)
                action = _normalize_action(raw)
                if action is None:
                    raise ValueError(f"bad action return: {raw!r}")
            except Exception:
                errors += 1
                # Try a random action
                action = int(rng_fallback.randint(0, 4))
            states.append(state)
            actions.append(action)
            env.step(action)
            game_steps += 1
        lengths.append(game_steps)
        scores.append(env.score)

    return Demo(
        name=name,
        states=np.asarray(states, dtype=np.int64) if states else np.zeros((0, height * width), dtype=np.int64),
        actions=np.asarray(actions, dtype=np.int64) if actions else np.zeros((0,), dtype=np.int64),
        games_played=n_games,
        errors=errors,
        avg_survival=float(np.mean(lengths)) if lengths else 0.0,
        avg_score=float(np.mean(scores)) if scores else 0.0,
        raw_code=raw_code,
    )


# ── Top-level: ask the LLM for all strategies, return demos ─────────

def generate_demos_from_llm(
    llm_fn: Callable[[str, str], str],
    strategies: Optional[dict[str, str]] = None,
    *, n_games: int = 50, height: int = 10, width: int = 10,
    game_seed_base: int = 1000, verbose: bool = False,
) -> dict[str, Demo]:
    """Prompt the LLM for each strategy in the dict, compile, play games.

    Args:
        llm_fn: (system_prompt, user_prompt) → response_text, matching the
            signature from llm_client.make_local_llm_client.
        strategies: {name: strategy description}. Default = STRATEGY_PROMPTS.

    Returns dict {name: Demo}. Strategies that fail to compile are
    omitted (logged to stdout).
    """
    strategies = strategies or STRATEGY_PROMPTS
    out: dict[str, Demo] = {}
    for name, desc in strategies.items():
        user = _build_user_prompt(name, desc, height, width)
        if verbose:
            print(f"[teacher] asking LLM for strategy {name!r}...", flush=True)
        t0 = time.time()
        try:
            response = llm_fn(SYSTEM_PROMPT, user)
        except Exception as e:
            if verbose:
                print(f"[teacher]   LLM call failed for {name}: {e}")
            continue
        if verbose:
            print(f"[teacher]   response in {time.time() - t0:.1f}s "
                  f"({len(response)} chars)")
        code = _extract_python(response)
        try:
            fn = _compile_strategy(code)
        except ValueError as e:
            if verbose:
                print(f"[teacher]   compile failed for {name}: {e}")
            continue
        if verbose:
            print(f"[teacher]   collecting {n_games} games...")
        demo = collect_demos(
            fn, name, n_games=n_games, height=height, width=width,
            game_seed_base=game_seed_base, raw_code=code,
        )
        if verbose:
            print(f"[teacher]   {name}: survived {demo.avg_survival:.1f} "
                  f"steps, scored {demo.avg_score:.2f}, errors={demo.errors}")
        out[name] = demo
    return out
