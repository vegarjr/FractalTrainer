# Sprint 15 / 16 / 17 — Options for self-improvement loop tests

**Recorded:** 2026-04-20, after v3 Sprint 14 (Snake refined negative).

Context: the Snake negatives (Sprints 13–14) revealed the **compose
primitive's scope boundary** (classification regime, not sequential
control). They did NOT invalidate the self-improvement machinery.
Three distinct self-improvement loops exist in this repo, and each has
a different test that would genuinely add information:

## Option 1 — v1 closed loop with local Qwen-7B

**Sprint 15a (~30 min of execution time).**

Run `scripts/run_closed_loop.py --llm local` to exercise the v1 hparam-
repair pipeline with the local Qwen-7B driving the LLM steps, instead
of the Claude CLI used in Sprint 3's original validation.

- **Why worth doing:** the local-LLM wiring (Sprint 12b precursor,
  Sprint 13 integrated `make_local_llm_client`) needs a real use in
  the repair loop to confirm Qwen-7B can produce valid
  `<<<PATCH>>> configs/hparams.yaml` blocks. Pure sanity-check.
- **Expected outcomes:** (a) patches accepted → local LLM good enough
  for hparam repair; (b) syntactically broken patches → may need
  tighter prompt engineering for Qwen; (c) valid YAML but hparams
  unchanged or worse → Qwen too conservative / lost.
- **Decision value:** tells us whether the full FractalTrainer
  pipeline can run entirely offline.

## Option 2 — v1 loop on a deliberately pathological starting config

**Sprint 15b (~1 day of tuning + analysis).**

Sprint 8's utility null was partly because the MNIST baseline was
already near-oracle. Pick a starting hparam config that trains badly
(e.g., lr=1.0, bs=1, no weight decay, no momentum) and test whether
the LLM-driven repair loop can meaningfully recover it vs. always
using a textbook default.

- **Why worth doing:** v2 Sprint 8's null ("golden-run loop ties
  textbook fix") may be an artefact of the healthy-baseline choice.
  Where the loop should shine is recovering from a bad state. This
  tests that.
- **Expected outcomes:** if the LLM loop recovers within N iterations
  to near-textbook quality, the repair value is real and
  demonstrable. If it doesn't, the null from Sprint 8 generalizes.
- **Decision value:** determines whether v1 has any practical
  application outside of diagnostics.

## Option 3 — v3 growth + v1 validation gates combined

**Sprint 15c (~1 week of architectural work).**

When the `FractalRegistry.decide()` method returns `spawn`, the
currently-spawned expert is trained with fixed hparams. Replace that
fixed-hparam training with a mini repair loop: the LLM proposes
spawn-time hparams scoped to the new task's label set, the three-gate
machinery (scope/schema/outcome) validates before committing, and the
trained expert's signature joins the registry only if outcome-gate
passes.

- **Why worth doing:** this is the combination neither v1 nor v3
  alone has tested. It's also the most vision-aligned: the LLM teaches
  the system HOW to train specialists, not just what to classify.
- **Expected outcomes:** per-task hparam tuning via LLM might give
  small improvements in behavior-clone / supervised settings. If large
  enough, it'd be a real productization direction.
- **Decision value:** defines whether the repair-loop infrastructure
  has a use case in the v3 architecture (currently it's vestigial).

## Prioritization

- **Option 1 first** — cheapest, clearest, produces immediate useful
  information.
- **Option 2 second if option 1 works** — extends the loop's use case.
- **Option 3 last** — meaningful only if options 1 and 2 show the
  infrastructure still has life in it.

---

## Why none of these is wasted time

Sprint 13–14's Snake negatives are specifically about the **compose
primitive** in sequential-control regimes. They do NOT say anything
about:

- The repair-loop's ability to edit config files safely (well-tested).
- The scope/schema/outcome gates (known correct).
- Hparam repair on supervised training targets (mechanically validated
  in Sprint 3).
- The v3 classification-regime registry (Sprints 9–12 validated).

So these self-improvement experiments target regimes where
FractalTrainer's pieces actually work — they aren't re-running the
failed Snake path.
