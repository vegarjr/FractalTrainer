"""v3 Sprint 15c-revised — Local LLM as data-describer for FractalTrainer.

New framing (course-correction from 15b): the local Qwen-7B isn't
good at proposing hparam repairs, but it's well-suited for DESCRIBING
incoming data. Instead of "fix this broken training," the local LLM
answers "what task am I looking at?" — which is exactly what
FractalTrainer's registry needs to route a new query.

The Sprint 7b finding (ρ = -0.848 between label-set Jaccard and
signature distance) means: if we know the incoming task's class-1
label set, we can predict which registry entries are relevant WITHOUT
computing a probe-batch signature. That maps cleanly onto a local
LLM's strengths — pattern matching on small labeled examples.

Pipeline tested here:
  1. For each of 5 "new" tasks (Sprint 7 subsets not in the target
     registry), sample 20 labeled examples (underlying MNIST digit
     + binary label 0/1).
  2. Prompt local Qwen: "Here are 20 (digit, label) pairs. What is
     the class-1 digit set?" Qwen outputs a set.
  3. Compare Qwen's identification to ground truth.
  4. Use Qwen's identification to look up the nearest registry entry
     by Jaccard overlap (metadata-only, no signature computation).
  5. Compare to signature-based routing (ground-truth) — does Qwen's
     described-set route to the same / a close entry?

Metric:
  a. Identification accuracy (exact set match / Jaccard overlap
     between Qwen's set and ground truth).
  b. Routing accuracy (does described-set-routing pick the same
     registry entry as signature-routing?).

If both are high, the local LLM is a useful perception layer: it
turns raw data into a structured description that the registry can
consume without any heavy signature computation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.repair.llm_client import (  # noqa: E402
    make_claude_cli_client,
    make_local_llm_client,
)


EXISTING_BINARY = [
    frozenset({1, 3, 5, 7, 9}),  frozenset({5, 6, 7, 8, 9}),
    frozenset({2, 3, 5, 7}),     frozenset({0, 1, 2, 3, 4}),
    frozenset({1, 3, 6}),        frozenset({1, 2, 3, 5, 8}),
    frozenset({4, 5, 6}),
]


def _sprint7_tasks() -> dict[str, frozenset[int]]:
    def _novel(s):
        full = frozenset(range(10))
        return all(s != ex and s != (full - ex) for ex in EXISTING_BINARY)

    cands = []
    for k in (3, 4, 5, 6):
        for c in itertools.combinations(range(10), k):
            if _novel(frozenset(c)):
                cands.append(tuple(sorted(c)))
    rng = random.Random(42)
    rng.shuffle(cands)
    return {
        "subset_" + "".join(str(d) for d in s): frozenset(s)
        for s in cands[:20]
    }


# ── Existing-registry task set (what we route into) ─────────────────

REGISTRY_TASK_NAMES = {
    "parity":          frozenset({1, 3, 5, 7, 9}),
    "high_vs_low":     frozenset({5, 6, 7, 8, 9}),
    "primes_vs_rest":  frozenset({2, 3, 5, 7}),
    "ones_vs_teens":   frozenset({0, 1, 2, 3, 4}),
    "triangular":      frozenset({1, 3, 6}),
    "fibonacci":       frozenset({1, 2, 3, 5, 8}),
    "middle_456":      frozenset({4, 5, 6}),
}
# Plus the 20 Sprint 7 subset_* tasks
REGISTRY_TASK_NAMES.update(_sprint7_tasks())


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


# ── Sample pairs from a task ────────────────────────────────────────

def sample_labeled_pairs(
    task_subset: frozenset[int], n_pairs: int,
    data_dir: str, seed: int,
) -> list[tuple[int, int]]:
    """Return n_pairs of (underlying_digit, binary_label). We peek at
    the true digit because the local LLM is text-only — it can't see
    images, so we show it the digit+label pairs directly. In a real
    multimodal system, the LLM would receive the image."""
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(base), size=n_pairs, replace=False)
    pairs = []
    for i in idx.tolist():
        _, digit = base[i]
        digit = int(digit)
        label = int(digit in task_subset)
        pairs.append((digit, label))
    return pairs


# ── LLM prompt ──────────────────────────────────────────────────────

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


def _build_user_prompt(pairs: list[tuple[int, int]]) -> str:
    lines = ["# Examples (digit, label):"]
    for d, l in pairs:
        lines.append(f"  ({d}, {l})")
    lines.append("\n# Output the class-1 set as a sorted JSON array.")
    return "\n".join(lines)


def _parse_set(text: str) -> Optional[frozenset[int]]:
    """Extract a JSON array of digits from the LLM response."""
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


# ── Main ────────────────────────────────────────────────────────────

def _make_llm(name: str, local_url: str, temperature: float):
    if name == "cli":
        return make_claude_cli_client()
    if name == "local":
        return make_local_llm_client(base_url=local_url,
                                      temperature=temperature,
                                      max_tokens=256)
    raise ValueError(name)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="local", choices=("local", "cli"))
    parser.add_argument("--local-llm-url",
                        default="http://127.0.0.1:8080")
    parser.add_argument("--local-temperature", type=float, default=0.3)
    parser.add_argument("--data-dir", default="results/data")
    parser.add_argument("--n-pairs", type=int, default=20,
                        help="examples shown to the LLM per task")
    parser.add_argument("--tasks", nargs="+",
                        default=["subset_267", "subset_01458",
                                 "subset_123689", "subset_459",
                                 "subset_0379"])
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--results-out",
                        default="results/data_describer.json")
    args = parser.parse_args(argv)

    all_tasks = _sprint7_tasks()
    test_tasks = {n: all_tasks[n] for n in args.tasks if n in all_tasks}

    print(f"Local-LLM data-describer ({args.llm})")
    print(f"Pairs shown per task: {args.n_pairs}")
    print(f"Tasks: {list(test_tasks.keys())}\n")

    llm_fn = _make_llm(args.llm, args.local_llm_url, args.local_temperature)

    rows: list[dict] = []
    for name, truth in test_tasks.items():
        pairs = sample_labeled_pairs(
            truth, args.n_pairs, args.data_dir, args.sample_seed)
        pos_digits = sorted({d for d, l in pairs if l == 1})
        neg_digits = sorted({d for d, l in pairs if l == 0})

        # Ask LLM
        user = _build_user_prompt(pairs)
        try:
            response = llm_fn(SYSTEM_PROMPT, user)
        except Exception as e:
            rows.append({"task": name, "truth": sorted(truth),
                          "error": str(e)})
            continue
        guess = _parse_set(response)

        # Identification metrics
        if guess is None:
            exact = False
            jac_id = 0.0
        else:
            exact = (guess == truth)
            jac_id = jaccard(guess, truth)

        # Routing — find nearest task in REGISTRY_TASK_NAMES by Jaccard
        # (a) with ground truth (oracle routing)
        oracle_candidates = [
            (jaccard(truth, REGISTRY_TASK_NAMES[n]), n)
            for n in REGISTRY_TASK_NAMES if n != name
        ]
        oracle_candidates.sort(reverse=True)
        oracle_route = oracle_candidates[0][1]
        oracle_jac = oracle_candidates[0][0]

        # (b) with Qwen's guess
        if guess is None or len(guess) == 0:
            llm_route = None
            llm_route_jac = 0.0
            route_match = False
        else:
            llm_candidates = [
                (jaccard(guess, REGISTRY_TASK_NAMES[n]), n)
                for n in REGISTRY_TASK_NAMES if n != name
            ]
            llm_candidates.sort(reverse=True)
            llm_route = llm_candidates[0][1]
            llm_route_jac = llm_candidates[0][0]
            route_match = (llm_route == oracle_route)

        row = {
            "task": name,
            "truth": sorted(truth),
            "n_positive_seen": len(pos_digits),
            "n_negative_seen": len(neg_digits),
            "positive_digits_in_sample": pos_digits,
            "negative_digits_in_sample": neg_digits,
            "llm_guess": sorted(guess) if guess is not None else None,
            "identification_exact": exact,
            "identification_jaccard": jac_id,
            "oracle_route_nearest": oracle_route,
            "oracle_route_jaccard": oracle_jac,
            "llm_route_nearest": llm_route,
            "llm_route_jaccard": llm_route_jac,
            "route_matches_oracle": route_match,
            "llm_response_preview": response[:200],
        }
        rows.append(row)
        flag_id = "✓" if exact else "✗"
        flag_rt = "✓" if route_match else "✗"
        print(f"  {name}  truth={sorted(truth)}  "
              f"guess={sorted(guess) if guess else 'None'}")
        print(f"    identification: {flag_id} (Jaccard "
              f"{jac_id:.3f})  |  "
              f"routing: {flag_rt} "
              f"(oracle→{oracle_route}, llm→{llm_route})")

    # ── Summary ──
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    n = len(rows)
    id_exact = sum(1 for r in rows if r.get("identification_exact"))
    route_match = sum(1 for r in rows if r.get("route_matches_oracle"))
    mean_jac_id = float(np.mean(
        [r.get("identification_jaccard", 0.0) for r in rows])) if rows else 0.0
    print(f"  Tasks tested:                 {n}")
    print(f"  Exact set identification:     {id_exact}/{n}")
    print(f"  Mean Jaccard(guess, truth):   {mean_jac_id:.3f}")
    print(f"  Routing matches oracle:       {route_match}/{n}")

    # Save
    out = {
        "llm": args.llm,
        "n_pairs": args.n_pairs,
        "rows": rows,
        "summary": {
            "n_tasks": n,
            "exact_id": id_exact,
            "mean_jaccard_guess_vs_truth": mean_jac_id,
            "routing_matches_oracle": route_match,
        },
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
