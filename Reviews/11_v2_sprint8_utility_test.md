# v2 Sprint 8 — Utility test: honest null on MNIST-MLP

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## The question

Sprints 1–7 proved the machinery works. Sprint 8 asks whether it's actually *useful* — does golden-run-guided repair outperform simpler baselines on held-out test accuracy?

## Setup

Four (plus one null) conditions, identical starting divergent config, identical MNIST train/test split (data_seed=42).

Starting config: `lr=0.1, adam, wd=0, do=0, seed=42`. Per the Sprint 2 science sweep, this is a known gradient-exploding Adam configuration.

| ID | Condition | Method |
|---|---|---|
| C0 | no_fix | Train from the divergent config as-is |
| C1 | golden_run_loop | `scripts/run_closed_loop.py --llm cli` against opaque golden → train with final hparams |
| C2 | first_aid_halve_lr | Halve the learning rate, keep everything else |
| C3 | textbook_adamw_small_lr | `AdamW, lr=1e-3, wd=1e-2, do=0.1` — standard "fix diverging Adam" recipe |
| C4 | random_search (n=5) | Sample 5 random hparam configs from a sensible prior, pick best test_acc |

All conditions train a 784→64→32→10 MLP on the same MNIST-5000 subset, evaluate on the same MNIST-1000 held-out subset.

## Results

```
condition                        test_acc    wall_s
------------------------------------------------------------
no_fix                             0.4140       9.4
golden_run_loop                    0.9220      91.3
first_aid_halve_lr                 0.7650       8.1
textbook_adamw_small_lr            0.9230       8.4
random_search                      0.9350      41.1
```

**Ranking by test_acc:** random_search > textbook ≈ golden_run_loop >> first_aid >> no_fix.

## Reading

### The system works, but doesn't win

The golden-run loop *does* fix divergent training (0.41 → 0.92 in test_acc — a +0.51 improvement). That's real.

But on this task it doesn't demonstrably outperform cheap alternatives:
- **Random search (5 points)** beats the loop by ~1 percentage point (0.935 vs 0.922), for about half the wall clock and **zero LLM calls**. The difference is within noise at a single evaluation, so "tied or slightly better" is fair.
- **Textbook "AdamW + small lr + regularization"** ties the loop exactly (0.923 vs 0.922), **10× faster** and no LLM.
- **Naive "halve the lr"** is meaningfully worse (0.765) — confirms the fix isn't trivial.

Claude's chosen final hparams for C1 (`lr=0.001 adam wd=0.001 do=0 seed=42`) are close to the textbook fix, which is why their test accuracies land nearly identical. The loop isn't adding value *beyond* textbook prior on this task — it's largely *re-deriving* textbook prior from the geometric signal.

### Caveats worth stating

1. **Single trial per condition.** n=1 per method; noise is real. Multi-seed error bars would probably wash most of these differences out — except the ones bigger than ~3 points (no_fix, first_aid).
2. **Random search got a lucky draw** (candidate 2 hit `lr=0.006 adamw wd=0.022 do=0.03` — good config). Range across 5 random samples was 0.276 to 0.935. With a different `--search-seed`, random search could have been much worse than the loop.
3. **MNIST-MLP has a very flat optimum landscape.** Many hparam combinations converge to ~92% test_acc. That makes this task a *hard venue* for demonstrating the utility of any sophisticated tuning method — most things work. A harder task (CIFAR-CNN, small transformer fine-tune) would stress the signal.
4. **The loop was limited to hparam patches.** Candidate B (training-code rewrites) would let the LLM change architecture/loss — potentially more valuable when hparams alone aren't enough. Untested.

### Interpretation

The system's validated status is:

- **Research tool: yes.** The machinery is sound. The meta-trajectory dim measurement (Sprint 7) is a genuinely novel observation. The 3-attractor structure of Claude's fixes is interesting.
- **Production tool for MNIST-MLP: no.** Random search or textbook advice is cheaper, tied-or-better in outcome, and has no dependency on an LLM. There's no utility case for the loop *on this task*.
- **Production tool on harder tasks: unknown.** The comparison could look very different on CIFAR, on imbalanced data, on transformer training where divergent behavior is less cookbook.

### Why the textbook baseline ties the loop

Looking at what Claude's loop actually did: it took `lr=0.1 adam` → `lr=0.001 adam wd=0.001`. That's the *textbook response* — reduce step size by 100×, add a touch of weight decay. Claude derived it from geometric deltas (step_norm_std high, path_length high) but arrived at essentially the canonical advice. On a task where the canonical advice is correct, the loop is an expensive way to re-derive it.

## Implications for the v2+ campaign

- **Candidates C (meta-recursion) and B (training-code rewrites) on MNIST-MLP** would both likely run into the same ceiling: the canonical advice on MNIST is strong, simple baselines are hard to beat, and the extra machinery can't produce a better config than "lr=0.001 adamw +reg".
- **Transfer to harder tasks** is the meaningful next frontier, but that's a bigger project — different datasets, different architectures, potentially different goldens per task.
- **The scientific observations** (Sprints 2, 5, 7) stand regardless of utility outcome.

## Decision

**v2 Sprint 8 gate: PASSED** (gate passes on "we honestly answered the question").

**Utility claim on MNIST-MLP: NULL.** The system works but doesn't win against cheap baselines on this task. That's a legitimate finding, not a failure — we didn't validate a utility we didn't actually have.

## Recommended next steps

If you want to keep pushing FractalTrainer:
- **Test on a harder task** (e.g., tiny ViT on CIFAR, or fine-tuning a small LM) to see if the loop outperforms baselines when textbook advice is weaker.
- **Multi-seed error bars** on the current comparison, to quantify noise.
- Write up the scientific findings (meta-trajectory dim, attractor structure) as the main contribution, with utility noted as task-dependent.

If you want to move on:
- **GeometricAI seed panels** (paper-finish-line work) — tasks #28–#32 are still pending and deliver concrete publishable progress.
- **Candidate B or C** — cool demonstrations, but this experiment suggests neither will deliver utility on MNIST that exceeds textbook advice.

## Paired ChatGPT review

Per the dual-AI workflow, this is Claude's honest null result. A ChatGPT-side read should validate (a) whether single-trial comparison is sufficient or multi-seed is required, and (b) whether the "MNIST is too easy" interpretation is reasonable or whether we should have picked a harder test bench from the start.
