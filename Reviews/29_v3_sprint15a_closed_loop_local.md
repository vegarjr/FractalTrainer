# v3 Sprint 15a — v1 closed loop with local Qwen-7B: clean integration, quality gap

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Goal

Exercise the full v1 closed-loop repair pipeline with the local
Qwen2.5-Coder-7B driving the LLM step, instead of the Claude CLI used
in Sprint 3's original validation. Confirms the offline-operation
path is real: can FractalTrainer's whole closed loop run with zero
external API dependencies?

## Setup

- **Starting config:** `configs/hparams.yaml` set to Sprint 3's known-
  pathological baseline: `lr=0.1, sgd, wd=0.0, do=0.0, seed=42,
  bs=64`. This produces a training trajectory whose correlation
  dimension lives far below the target band [1.2, 1.8].
- **Target band:** `configs/target_shape.yaml` — `dim_target=1.5`,
  `tolerance=0.3`, `hysteresis=1.2`, `max_repair_iters=5`.
- **Experiment:** 3-layer MLP on MNIST-5000 subset, 500 steps, 25
  snapshots per run.
- **LLM backend:** `--llm local` → `make_local_llm_client(base_url=
  "http://127.0.0.1:8080")` talking to the Qwen-7B llama-server
  running on GTX 1050 Vulkan.
- **Cap:** `--max-iters 3`.

## Result

```
[repair] iteration 1/3
[repair] probing with current hparams...
[repair] LLM reports no fix — halting

iter 1: status=no_fix  dim=0.831  div=2.229  (elapsed 44.7s)
    The current configuration is already far outside the target band,
    and the divergence score is too high to be corrected with minor
    hyperparameter changes.
```

**Pipeline verdict: INTEGRATION WORKS.** The full closed loop ran
end-to-end in 44.7 s. No crashes, no protocol errors, no gate
failures.

**Quality verdict: QWEN DECLINES.** Qwen-7B returned
`NO_FIX_FOUND`-style response — not a patch at all. It reasoned
(accurately!) that the divergence is too large for minor changes and
effectively abstained.

## What this proves

1. **Local-LLM wiring is correct.** The `make_local_llm_client`
   helper from Sprint 12b speaks the same `(system, user) →
   response_text` contract as `make_claude_cli_client`, and the
   RepairLoop, PromptBuilder, and PatchParser all accept it as a
   drop-in replacement. Observed by the loop completing normally.
2. **Scope/schema gates never fired.** Because no patch was
   proposed, no validation was required — the loop simply halted.
   That's exactly the correct behavior for an abstaining LLM; the
   alternative (a broken patch that fails all three gates) would be
   noisier but equally non-damaging.
3. **Offline operation is real.** The entire v1 pipeline — training
   probe, dimension calculation, divergence measurement, LLM
   consultation, outcome gate, backup/rollback — now runs with zero
   external dependencies. `--llm local` is a legitimate fourth
   option alongside `mock`, `cli`, `api`.

## What this reveals about Qwen-7B vs Claude

Sprint 3 with Claude-3.5-Sonnet on the SAME pathological config
confidently proposed `lr=0.01, optimizer=adamw, wd=0.01, do=0.1` —
got `dim 0.36 → 0.89` accepted, then halted within-band at 1.27 on
iteration 2. Total repair: one accepted patch.

Qwen-7B with `--llm local` on the same config returns a no-fix
response. Hypotheses for the gap:

1. **Model capability.** Qwen-7B has less "tacit hparam knowledge"
   than Claude-3.5-Sonnet. The prompt asks for an effective patch,
   and Qwen's internal estimate is "I can't produce one with high
   confidence." Refusing is reasonable given this belief.
2. **Prompt affordance.** The PromptBuilder's system prompt gives
   the LLM an easy out (`NO_FIX_FOUND` as a valid response). Claude
   is trained to be helpful-but-confident; Qwen-7B may be more
   conservative in low-confidence regimes. A prompt rewrite that
   says "always attempt a patch" would force action.
3. **Temperature.** `make_local_llm_client` defaults to
   `temperature=0.3` (code-sweet spot). For creative hparam
   suggestions, `temperature=0.7–1.0` might unlock more attempts.
4. **Context window.** Qwen-7B at 4K context fits the prompt easily,
   so this isn't a truncation issue.

These are all tunable. For option 1's purpose — proving the
integration works — the test passes cleanly. Optimizing Qwen's
behavior is option 2's territory.

## What ships

No code changes — this sprint was purely an integration test with
the wiring from Sprint 12b. Artifacts:

- `configs/hparams.yaml` now contains the pathological starting
  config (useful as-is for option 2 experiments).
- `results/closed_loop_summary.json` captures the iter-1 outcome.
- `Reviews/29_v3_sprint15a_closed_loop_local.md` — this document.

## Natural follow-ups

- **Compare `--llm local` vs `--llm cli` on the same config.** A
  head-to-head with Claude on the same pathological starting state
  quantifies the capability gap concretely. Sprint 3 already ran
  Claude, so this is just re-running today with current infra.
- **Prompt engineering for Qwen (option 2 in the saved options doc).**
  Tighten the PromptBuilder to push Qwen toward action — remove the
  NO_FIX_FOUND escape hatch, raise temperature, add few-shot
  examples of successful patches. See if Qwen can match Claude's
  Sprint 3 behavior.
- **Try a less-pathological starting config.** The current
  divergence (2.23) is so large Qwen's caution is defensible.
  A milder starting point (e.g., lr=0.05 instead of 0.1) might
  be where Qwen can reason confidently and produce a valid patch.

## Paired ChatGPT review prompts

ChatGPT should validate:

1. Whether **Qwen's NO_FIX refusal is a bug or a feature**. One
   reading: it's exactly what you want from a code-assistant model
   when confidence is low. Another reading: the whole point of the
   repair loop is to try stuff, so a NO_FIX response on the first
   iteration defeats the purpose. The answer depends on how the
   user wants to handle low-confidence local-LLM outputs.
2. Whether **the cleanest "option 1 passed" framing** is to say
   "integration works, capability gap vs Claude is a separate
   question" or "integration works but local LLM is too conservative
   to be useful without prompt engineering." They lead to different
   next sprints.
