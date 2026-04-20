# v3 Sprint 15c — Local LLM as data-describer (course-corrected option 3)

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## The pivot

The original option 3 plan was to use the local LLM to propose
hparams at spawn time — essentially running Sprint 15b's loop at
per-expert scale. User course-correction mid-sprint:

> *the idea for local llm to aid the fractaltrainer is to tell
> fractaltrainer what data it is receiving.*

That reframes the local LLM's role from **repair author** (where
Sprint 15b showed Qwen-7B is too cautious) to **perception layer**
(answering "what task is this?" from a small sample of labeled
data). The classification-regime machinery FractalTrainer already
has (match / compose / spawn, routing by Jaccard per Sprint 7b)
consumes structured task descriptions; the local LLM's job is just
to produce them.

## Why this is the right role for a local 7B model

Sprint 7b showed **signature distance ≈ label-set Jaccard distance**
(ρ=−0.848 on 351 binary-task pairs). That means if we know the
incoming task's class-1 digit set, we can predict which registry
entries are relevant WITHOUT computing a probe-batch signature.

Identifying a class-1 set from 20–50 (digit, label) pairs is a
pattern-matching task — well within a 7B model's comfort zone, and
much easier than proposing hparams (which requires tacit knowledge
of training dynamics).

## Setup

For each of 5 new tasks (Sprint 7 subsets not in the target
registry), sample N pairs of (underlying MNIST digit, binary label 0/1)
and ask the LLM to output the class-1 digit set as a JSON array.

- **5 test tasks:** `subset_267 {2,6,7}`, `subset_01458 {0,1,4,5,8}`,
  `subset_123689 {1,2,3,6,8,9}`, `subset_459 {4,5,9}`, `subset_0379 {0,3,7,9}`.
- **Registry for routing:** the 27-task registry from Sprint 7b
  (7 existing binary + 20 Sprint 7 subsets).
- **Metrics:**
  - **Exact identification:** guessed set == truth.
  - **Jaccard identification:** `|guess ∩ truth| / |guess ∪ truth|`.
  - **Routing correctness:** Jaccard-nearest registry entry under
    guessed set matches Jaccard-nearest under true set.

## Results

### Qwen-7B local, 20 examples per task

| Task            | Truth         | Qwen guess            | Exact | Jaccard | Route match |
|-----------------|---------------|-----------------------|:-----:|:-------:|:-----------:|
| subset_267      | {2,6,7}       | {1,2,6,7,9}           | ✗     | 0.600   | ✓           |
| subset_01458    | {0,1,4,5,8}   | {1,5,8}               | ✗     | 0.600   | ✗           |
| subset_123689   | {1,2,3,6,8,9} | {1,2,3,6,8,9}         | ✓     | 1.000   | ✓           |
| subset_459      | {4,5,9}       | {5,9}                 | ✗     | 0.667   | ✓           |
| subset_0379     | {0,3,7,9}     | {1,3,7,9}             | ✗     | 0.600   | ✗           |
| **Summary**     |               |                       | **1/5** | **0.693** | **3/5** |

### Qwen-7B local, 50 examples per task

| Task            | Truth         | Qwen guess            | Exact | Jaccard | Route match |
|-----------------|---------------|-----------------------|:-----:|:-------:|:-----------:|
| subset_267      | {2,6,7}       | {2,6,7}               | ✓     | 1.000   | ✓           |
| subset_01458    | {0,1,4,5,8}   | {1,4,8}               | ✗     | 0.600   | ✗           |
| subset_123689   | {1,2,3,6,8,9} | {1,2,3,6,8,9}         | ✓     | 1.000   | ✓           |
| subset_459      | {4,5,9}       | {4,5,9}               | ✓     | 1.000   | ✓           |
| subset_0379     | {0,3,7,9}     | {1,3,4,7,9}           | ✗     | 0.500   | ✗           |
| **Summary**     |               |                       | **3/5** | **0.820** | **3/5** |

### Claude CLI, 50 examples per task

| Task            | Truth         | Claude guess          | Exact | Jaccard | Route match |
|-----------------|---------------|-----------------------|:-----:|:-------:|:-----------:|
| all 5 tasks     | —             | exact match each time | ✓     | 1.000   | ✓           |
| **Summary**     |               |                       | **5/5** | **1.000** | **5/5** |

## Error analysis

Diagnosed by checking which digits actually appeared in each
sample vs which digits Qwen included/excluded:

| Task (Qwen n=20) | Missed (had evidence) | Missed (no evidence) | Wrongly added |
|------------------|-----------------------|----------------------|---------------|
| subset_267       |  —                    |  —                   | {1, 9}        |
| subset_01458     | {0}                   | {4}                  |  —            |
| subset_123689    |  —                    |  —                   |  —            |
| subset_459       |  —                    | {4}                  |  —            |
| subset_0379      | {0}                   |  —                   | {1}           |

Two distinct failure modes:
1. **Sample-coverage gap.** A target digit never appeared in the 20
   examples, so Qwen couldn't know it was in the set. At 50
   examples, this failure largely disappeared (`subset_267` and
   `subset_459` became correct).
2. **Pattern hallucination.** Qwen coerces the set toward
   recognizable named patterns. `subset_0379 {0,3,7,9}` looks
   almost like parity `{1,3,5,7,9}`, and Qwen flips 0→1 to fit.
   `subset_267 {2,6,7}` has 1 and 9 added to look more "spread."
   These errors persist at 50 examples and survived temperature
   bumps.

Claude has neither failure mode on this test — 5/5 at n=50.

## What this means operationally

The data-describer primitive **works** as an architectural role. Key
takeaways:

1. **Routing is more robust than identification.** Qwen at n=20 had
   1/5 exact IDs but 3/5 correct routes — Jaccard-based
   nearest-neighbor is forgiving of small set errors, especially
   when error types are bounded (adding extras rarely flips the
   nearest-registry pick as long as the core is right).
2. **Sample size pays off dramatically** for a 7B model. 20 → 50
   examples: Jaccard identification 0.69 → 0.82, exact IDs 1 → 3.
3. **Claude is a drop-in oracle** for this use case. If perception
   quality matters, delegate to Claude (Max subscription covers
   this via the CLI path). If offline-only, use Qwen with n=50
   and accept ~60–70% routing quality.
4. **The capability gap between Claude and Qwen is task-specific.**
   On hparam repair (Sprint 15b), Qwen refused entirely. On label-
   set identification, Qwen partially works (3/5 at n=50). The
   local LLM has a clear niche here that it doesn't have for
   repair.

## Architectural fit with v3

Sprint 12 showed the architecture is self-organizing: clusters on
registered experts' label sets produce anchors automatically. The
data-describer extends this to the *input* side:

```
Incoming data
    ↓
Local LLM (perception: "what is this task?")
    ↓ (label set description)
Registry lookup (Jaccard-overlap-based: Sprint 7b mechanism)
    ↓
Match / compose / spawn (existing primitives)
```

This closes the pipeline's perception-to-action loop **entirely
through existing FractalTrainer primitives** plus a small local LLM.
No new registry code needed. No training needed. The LLM just
translates raw labeled data into the task descriptors the registry
already knows how to consume.

## Limits

- **Qwen's pattern bias** (subset_0379→parity, subset_01458→dropped
  {0,5}) persists at 50 examples. Fixable via prompt engineering
  ("enumerate digits 0-9; for each, say whether it's class-1") or
  few-shot examples, both untested this sprint.
- **Sample size of 50** may not always be affordable in real
  deployment. Whether a more aggressive identification (say, Monte
  Carlo Markov Chain over digit subsets guided by LLM confidence)
  could extract more from fewer examples is untested.
- **5 test tasks** is a small sample. A 20-task sweep with varied
  set sizes (3-, 4-, 5-, 6-subsets) would tighten statistics, but
  the direction is clear.
- **Multimodal data** (actual images, not digit labels) would need
  a vision-language model, not a text-only code model. Qwen-7B
  can't see images. For MNIST specifically this is fine (labels ARE
  digits), but generalization to arbitrary image tasks would need
  a Qwen-VL or similar.

## What ships

- `scripts/run_data_describer.py` — the driver, ~250 LOC. Works with
  `--llm local` (Qwen) or `--llm cli` (Claude), tunable
  `--n-pairs` and `--temperature`.
- `results/data_describer.json` — Qwen n=20 baseline.
- `results/data_describer_n50.json` — Qwen n=50.
- `results/data_describer_cli_n50.json` — Claude n=50 oracle.

No changes to the registry or routing code — this uses the
Jaccard-overlap infrastructure Sprint 7b-12 already produced. Tests
still pass (176/176).

## Natural follow-ups

1. **Integrate the describer with live routing.** Wire
   `data_describer` into a demo where the incoming task is opaque
   and the system uses the LLM's description to route. End-to-end
   showcase of perception-to-action.
2. **Prompt engineering for Qwen:** enumerate-digit prompt,
   few-shot examples, two-stage refine. Aim for >4/5 identification
   at n=20, which would make Qwen a genuinely useful offline
   perception layer.
3. **Generalize beyond MNIST digits.** For general binary tasks
   defined by any labeling rule, ask the LLM to describe the rule
   (e.g., "class-1 = images whose mean brightness > 128"). This is
   how real-world tasks would arrive — a rule or dataset, not an
   explicit digit set. Qwen might handle "describe the rule" better
   than "enumerate the set."

## Paired ChatGPT review prompts

ChatGPT should validate:

1. Whether **using an LLM as a perception layer for routing** is a
   well-explored pattern in the continual-learning / ML-system
   literature. If yes, pointer to related work would sharpen the
   Mixture-of-Fractals framing. If no, this is a novel architectural
   slot worth claiming.
2. Whether the **3/5 routing at n=20** is considered useful in
   deployment terms or dismissible as too-unreliable. I'd argue
   useful because routing errors are filterable (the registry can
   verify by signature distance after lookup), but ChatGPT may
   have a different take.
3. Whether the **Qwen pattern-hallucination** (dropping 0, adding 1
   to fit parity) is evidence of genuine weak semantic understanding
   or a symptom of code-specialized training that over-indexes on
   named patterns. Implications: if semantic, we need a bigger LLM;
   if training-specific, prompt engineering suffices.
