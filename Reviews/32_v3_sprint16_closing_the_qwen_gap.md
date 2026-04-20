# v3 Sprint 16 — Closing the Claude-Qwen gap (paths 1, 2, 3)

**Date:** 2026-04-21
**Reviewer:** Claude Opus 4.7 (1M context)

## Goal

Sprint 15b established that Qwen2.5-Coder-7B (local, Vulkan) refuses
to drive the v1 repair loop on pathological hparam configs where
Claude-3.5-Sonnet succeeds. Three paths to close the gap were
enumerated:

1. **Fine-tune Qwen** on repair-loop demonstrations.
2. **Prompt-engineer** — few-shot patch examples.
3. **Self-ensemble** — sample Qwen N times at different temperatures.

Sprint 16 runs all three.

## Sprint 16a — Few-shot prompt (path 2)

**Change:** `PromptBuilder(include_fewshot=True)` prepends 3 worked
patch examples to the user prompt, showing common failure-mode
remedies:
- lr=0.1+sgd → lr=0.001+adam+dropout (underfitting)
- no regularization + overdispersed → add wd=0.05 + dropout=0.4 (overfit)
- lr=1e-5 (too tiny) → lr=0.005 (underfit)

`scripts/run_closed_loop.py --fewshot` enables this.

**Result:** Qwen still returns `NO_FIX_FOUND` on the pathological
config. Response word-for-word near-identical to Sprint 15b:

```
"The current model's trajectory is far from the target fractal
dimension and the divergence score is too high. A small change to
hparams may not be sufficient to achieve the desired target."
```

Despite seeing 3 worked patches that ARE "large changes"
(lr 0.1 → 0.001 is an order-of-magnitude move), Qwen's prior that
"large divergence requires small changes" overrode the demonstrations.

## Sprint 16b — Temperature-diversity self-ensemble (path 3)

**Setup:** `scripts/run_closed_loop_ensemble.py` wraps the LLM: for
each iteration, sample at 5 temperatures `{0.2, 0.5, 0.7, 0.9, 1.2}`,
return the first response that parses as a valid patch. Combined
with `--fewshot`.

**Result:** 5/5 samples returned `NO_FIX_FOUND` on the same pathological
config. All five responses were near-identical refusals. Hypothesis
was that sampling diversity would occasionally produce a patch by
chance; it didn't. Qwen's refusal is **prompt-deterministic** on
this problem regardless of temperature.

## Sprint 16c — Fine-tune dataset generation (path 1)

**Setup:** `scripts/generate_qwen_finetune_data.py` runs Claude CLI
on N pathological configs, captures every accepted iteration's
`(system_prompt, user_prompt, claude_response)` triple, and writes a
JSONL in OpenAI chat-message format suitable for Qwen SFT:

```json
{"messages": [
    {"role": "system",    "content": "<repair system prompt>"},
    {"role": "user",      "content": "<iteration's user prompt>"},
    {"role": "assistant", "content": "<<<PATCH\nfile: configs/hparams.yaml\n..."}
 ],
 "metadata": {"config_name": "lr_too_high_sgd", "iteration": 2,
               "dim_before": 0.77, "dim_after": 1.33,
               "divergence_before": 2.45, "divergence_after": 0.58}}
```

**Result:** 3 configs attempted → 2 accepted-iteration examples
captured. `lr_too_high_sgd`: Claude's iter-2 patch (lr 0.1 → 0.01,
bs 64 → 16, optimizer sgd → adam) moved dim 0.77 → 1.33 (within
band). `no_regularization_large_batch` converged in iter 1 (no LLM
proposal needed). `lr_too_low`: Claude proposed patches but outcome
gate rejected all (divergence didn't improve) — 0 accepted
examples.

The dataset pipeline works correctly. 2 examples is nowhere near
enough to actually fine-tune — for a real LoRA you'd want
100–1000+ demonstrations — but the plumbing is validated and
reusable. Running on 50+ configs with diverse pathology would
produce a training-scale dataset in a few hours.

## Combined verdict

| Path | Result | Cost | Unlocks Qwen? |
|------|--------|-----:|:--------------|
| 2 (few-shot) | Qwen still refuses | free | **No** |
| 3 (5-temp ensemble + few-shot) | 5/5 refuse | free | **No** |
| 1 (fine-tune dataset) | 2 Claude examples captured, pipeline works | ~3 min of Claude CLI calls | **Unknown (fine-tune not run here)** |

**Conclusion:** Qwen-7B Q4_K_M has a **prompt-deterministic refusal
mode** on this task. No surface-level intervention (prompt
engineering, sampling diversity, few-shot examples) penetrates it.
**The only remaining path is actual fine-tuning** on Claude-generated
demonstrations, which this machine's 4 GB VRAM can't do.

This aligns with the capability-theory framing from the earlier
discussion: scale + calibration + alignment are the axes where
Claude has advantages a 7B-via-prompting can't close. Fine-tuning
is the only lever that can actually move these — particularly
calibration (when-to-try-vs-refuse) — via RLHF-style training on
(prompt, helpful-response) pairs.

## Next-step options (not executed here)

1. **Fine-tune Qwen-7B on cloud GPU** (A100 or equivalent). The
   dataset pipeline in `generate_qwen_finetune_data.py` is ready;
   expand from 3 configs → 50 configs → ~100+ training examples.
   Then ~30 min on A100 for a LoRA SFT. Expected to reduce (maybe
   eliminate) the refusal-rate on seen pathological patterns.

2. **Different local model** — Qwen2.5-Coder-14B or DeepSeek-Coder-V2-
   Lite. Both need more than 4 GB VRAM. A used RTX 3060 12GB would
   unlock 14B; a 24GB card (3090/4090) unlocks 32B+.

3. **Accept the hybrid split** — Claude drives repair, Qwen handles
   perception (Sprint 15c's data-describer role, where Qwen IS
   useful). This is the pragmatic answer that the evidence supports:
   repair loops need stronger models; perception works locally.

## What ships

- `src/fractaltrainer/repair/prompt_builder.py` — `PromptBuilder
  (include_fewshot=True)` option + 3 hard-coded worked examples.
- `src/fractaltrainer/repair/repair_loop.py` — `RepairLoop
  (include_fewshot=...)` parameter threaded through.
- `scripts/run_closed_loop.py` — `--fewshot` flag added.
- `scripts/run_closed_loop_ensemble.py` — new, 5-temperature wrapper.
- `scripts/generate_qwen_finetune_data.py` — new, Claude → JSONL.
- `results/closed_loop_ensemble.json` — 16b result.
- `results/qwen_finetune_data.jsonl` — 2 example demonstrations (seed
  dataset; expand before actual fine-tuning).
- `results/qwen_finetune_trace.json` — full Claude run traces.
- `Reviews/32_v3_sprint16_closing_the_qwen_gap.md` — this doc.

176/176 tests still pass. The few-shot prompt change is a
non-breaking addition (default behavior unchanged when
`include_fewshot=False`).

## Paired ChatGPT review prompts

1. Whether **prompt-deterministic refusal across five temperature
   samples and few-shot examples** is evidence of genuine reasoning-
   capability limits, or just evidence that Qwen was instruction-
   tuned toward conservative behavior on uncertain decisions. The
   two have different implications for fine-tuning strategy (more
   examples vs. explicit anti-refusal RLHF).
2. Whether the **2-example dataset is publishable as a seed** for a
   community-fine-tuned "repair-loop Qwen" variant, or whether 50+
   examples is the minimum threshold worth sharing. My instinct:
   ~50 examples is the floor for a meaningful SFT experiment.
3. Whether the **"accept the hybrid split" outcome** is a
   satisfying resolution of the local-LLM question, or whether
   cloud fine-tuning is a cleaner deliverable before closing out
   the offline-operation thread.
