"""Describer adapter — refactor Sprint 15c's data-describer into a reusable class.

The describer turns raw (digit, binary_label) pairs into a frozenset of
class-1 digits, which the router consumes via Jaccard distance to
registry tasks. Two implementations:

    Describer(llm_fn)     — calls a language model (Claude CLI, Qwen, API)
    MockDescriber(answer) — returns a fixed frozenset; used for CI tests
                             and for `--llm mock` mode in the demo, where
                             the describer output is the ground-truth
                             label set (so routing behavior is isolated
                             from LLM noise).

System prompt is lifted verbatim from `scripts/run_data_describer.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol


SYSTEM_PROMPT = """You identify the labeling rule of a new binary
classification task on MNIST digits. You see 20 examples as (digit,
label) pairs, where label is 0 or 1. Your job: output the set of
digits that get label 1 (the 'class-1 set').

Rules:
 - Output ONLY a JSON array of digits, sorted ascending, no prose.
   Example: [0, 4, 5, 9]
 - Infer the class-1 set from the visible examples. If ambiguous
   (some digits 0-9 don't appear in the 20 examples), give your
   best inference.
 - Do not include digits outside 0-9. Do not include the digit 0 if
   you think label=0 means "not class-1".
"""


@dataclass
class DescribeResult:
    """Output of Describer.describe.

    Attributes:
        guess: frozenset of class-1 digits inferred from the pairs.
            None when the LLM response didn't parse.
        raw_response: the LLM's raw text output (truncated to 500 chars).
        positive_digits_seen: digits that appeared with label=1 in the input.
        negative_digits_seen: digits that appeared with label=0 in the input.
    """

    guess: Optional[frozenset[int]]
    raw_response: str
    positive_digits_seen: list[int]
    negative_digits_seen: list[int]


class LLMFn(Protocol):
    def __call__(self, system: str, user: str) -> str: ...


def build_user_prompt(pairs: list[tuple[int, int]]) -> str:
    lines = ["# Examples (digit, label):"]
    for d, l in pairs:
        lines.append(f"  ({d}, {l})")
    lines.append("\n# Output the class-1 set as a sorted JSON array.")
    return "\n".join(lines)


def parse_set(text: str) -> Optional[frozenset[int]]:
    m = re.search(r"\[[\s,\d]*\]", text)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    digits = set()
    for x in arr:
        try:
            d = int(x)
            if 0 <= d <= 9:
                digits.add(d)
        except (ValueError, TypeError):
            pass
    return frozenset(digits)


class Describer:
    """LLM-backed describer: (pairs) → frozenset[int]."""

    def __init__(self, llm_fn: LLMFn):
        self.llm_fn = llm_fn

    def describe(self, pairs: list[tuple[int, int]]) -> DescribeResult:
        pos = sorted({d for d, l in pairs if l == 1})
        neg = sorted({d for d, l in pairs if l == 0})
        user = build_user_prompt(pairs)
        response = self.llm_fn(SYSTEM_PROMPT, user)
        guess = parse_set(response)
        return DescribeResult(
            guess=guess, raw_response=response[:500],
            positive_digits_seen=pos, negative_digits_seen=neg,
        )


class MockDescriber:
    """Returns a fixed frozenset regardless of input.

    Used when we want to isolate routing behavior from LLM noise — the
    demo's `--llm mock` mode passes the task's ground-truth label set
    in here, so the pipeline's routing decision reflects only the
    signature-space behavior.
    """

    def __init__(self, answer: frozenset[int] | set[int] | list[int]):
        self.answer = frozenset(int(x) for x in answer)

    def describe(self, pairs: list[tuple[int, int]]) -> DescribeResult:
        pos = sorted({d for d, l in pairs if l == 1})
        neg = sorted({d for d, l in pairs if l == 0})
        return DescribeResult(
            guess=self.answer,
            raw_response=f"[mock] {sorted(self.answer)}",
            positive_digits_seen=pos, negative_digits_seen=neg,
        )


class OracleDescriber:
    """Returns the ground truth for whichever task the caller passes.

    Used by the demo when `--llm mock` is set and the scripted scenario
    gives a different ground-truth label set per query. Instead of
    constructing a separate MockDescriber per query, the pipeline holds
    one OracleDescriber and sets `current_truth` before each describe()
    call.
    """

    def __init__(self):
        self.current_truth: frozenset[int] = frozenset()

    def set_truth(self, truth: frozenset[int] | set[int] | list[int]) -> None:
        self.current_truth = frozenset(int(x) for x in truth)

    def describe(self, pairs: list[tuple[int, int]]) -> DescribeResult:
        pos = sorted({d for d, l in pairs if l == 1})
        neg = sorted({d for d, l in pairs if l == 0})
        return DescribeResult(
            guess=self.current_truth,
            raw_response=f"[oracle] {sorted(self.current_truth)}",
            positive_digits_seen=pos, negative_digits_seen=neg,
        )
