# Predecessor: AI_gpt_grok (March 2025)

This folder preserves the **intellectually necessary parts** of
`/home/vegar/AI_gpt_grok/` — a four-agent self-improving AI loop that
the user built in early March 2025, a few weeks before FractalTrainer.
It is the direct ancestor of FractalTrainer's repair architecture.

The original folder contained plaintext API keys and personal
credentials (see "Security note" at the bottom). This archive is
sanitized: `config.template.yaml` replaces the original `config.yaml`;
`love_message.txt` / `send.py` from the source are not copied; the
2 MB `ai_log.txt` and 85 KB `ai_memory.json` are sampled rather than
copied in full.

## Contents

```
predecessor_ai_gpt_grok/
├── README.md                         — this document
├── gpt_grok_ai.py                    — the core 540-line orchestrator (source unchanged)
├── config.template.yaml              — schema only, no keys
├── requirements.txt                  — original (incomplete — see lesson #4)
├── knowledge_base_excerpt.json       — cached search results showing
│                                        the repeated `exec()` errors
├── ai_memory_excerpt.json            — first few task entries from
│                                        the ~85 KB memory file
├── ai_log_excerpt_first200.txt       — first 200 lines of ai_log.txt
│                                        showing the openai-migration failure pattern
└── task_examples/
    ├── snake_modern.py               — "self_improvement_code.py"
    │                                    in original (misnamed — it's a snake game)
    ├── snake_replica_iter1_of_10_duplicates.py
    │                                  — one of 10 byte-identical files
    │                                    produced by the iteration loop
    ├── house_turtle.py               — "Hus.py", truncated at line 98
    ├── planets_3d.py                 — 3D orbit simulation
    ├── flappy_bird.py                — pygame Flappy Bird clone
    └── check_gpu.py                  — 4-line torch CUDA check
```

## What AI_gpt_grok was

A Python/Tkinter desktop app orchestrating **four LLM agents** in a
self-improvement loop:

| Agent       | Model                    | Role                         |
|-------------|--------------------------|------------------------------|
| Agent 1     | OpenAI GPT-4o            | Planner (break task into steps) |
| Agent 2     | xAI grok-2-latest        | Generator (write Python code)    |
| Agent 3     | Anthropic claude-3-5-sonnet | Refiner (clean up and fix)    |
| Agent 4     | Google Custom Search     | Researcher (find solutions online) |

The executor ran generated code via `subprocess.Popen` with a 10-second
timeout, captured stdout/stderr, and fed any failure back into the
planner for the next iteration. Max 100 iterations per task by default,
or infinite with `max_tries=0`.

Two special capabilities:

1. **Self-modification.** If a user's task string contained the literal
   substring `"improve your own code"`, the loop would back up
   `gpt_grok_ai.py` to `backups/gpt_grok_ai_backup_<timestamp>.py`,
   overwrite `gpt_grok_ai.py` with Grok-generated + Claude-refined
   replacement code, and restart itself with `subprocess.Popen(...)` +
   `sys.exit(0)`.

2. **Task-specific success criteria.** A hardcoded `is_doom_replica()`
   check looked for five literal substrings (`pygame`, `wad`, `raycast`,
   `enemy`, `weapon`) to decide if a Doom task was done. This was
   bolted on when the loop wouldn't terminate for general tasks.

## Lineage — what FractalTrainer kept and what it fixed

| AI_gpt_grok (March 2025)                        | FractalTrainer v3 (April 2025)                     |
|-------------------------------------------------|----------------------------------------------------|
| Four-agent orchestration (plan → gen → research → refine) | Three-gate repair (scope / schema / outcome)       |
| `shutil.copy` → backup before rewrite           | `configs/hparams.yaml` as the only patch target, git-reversible |
| Hard-coded per-task feature checks              | Geometric target band + golden-run match           |
| "Try until max_iters" (no principled stop)      | Triage rule: `if N<100 compose, else spawn`        |
| Tasks accumulated in `tasks/` by timestamp      | `FractalRegistry` keyed by activation signature    |
| Older task code passed as `current_code`        | Coverage-greedy ensemble selection from top-K      |
| No composition across tasks                     | Compose / match / spawn with content-aware routing |

The FractalTrainer `src/fractaltrainer/repair/` vendored modules
(`patch_parser.py`, `llm_client.py`, `prompt_builder.py`,
`repair_loop.py`) come from the `Fractal/evolution/stage11_introspection/`
repo — which itself evolved from ideas in this one. The direct
code-level link is indirect; the conceptual link is direct.

## Three lessons that shaped FractalTrainer

### 1. Without a principled stop criterion, loops don't converge

AI_gpt_grok's `self_improving_ai_loop` iterated until `max_tries` or
until a task-specific ad-hoc check passed. For the Snake task that
succeeded on iteration 1, the loop kept going and produced **ten
byte-identical 3226-byte files** in `tasks/` (you can see one of them
as `task_examples/snake_replica_iter1_of_10_duplicates.py`), because
the `break` after "success" was gated on `not doom_complete`, and the
feedback-for-next-iteration block ran before the break check.

FractalTrainer v3's triage rule (`if N < 100: compose; else: spawn`
per Sprint 9b; oracle beats it by only +1.1 pp per Sprint 9d) is the
quantitative stop this predecessor needed. The whole v3 arc — growth
threshold, match threshold, curated region anchor, coverage
redundancy — is one long answer to "when should the loop stop?"

### 2. Self-rewrite without validation corrupts itself silently

The `backups/` folder in the original has four timestamped snapshots:
15 199 bytes → 1 034 bytes → 1 633 bytes → 15 536 bytes. The two
middle backups are 6-10% of the surrounding file size, which strongly
suggests **at least one self-rewrite replaced `gpt_grok_ai.py` with
an almost-empty file** and the restart-and-iterate loop then wrote a
mostly-fresh version back. There is no `ast.parse()` check, no
smoke-test execution, no "does the new file still have a `main()`"
gate. The only safety is the timestamped backup copy, which means you
can recover by hand after a bad rewrite — not prevent one.

FractalTrainer v3's three gates are the fix:

- **Scope gate**: `patch_parser.validate_patches(allowed_files=["configs/hparams.yaml"])` rejects edits to anything but the hparam config.
- **Schema gate**: `hparam_config.validate_hparams()` rejects bad values, unknown keys, bad optimizer strings.
- **Outcome gate**: `shutil.copy2` backup, apply, re-measure divergence. Roll back if divergence_new > divergence_old or if loss exploded.

The pattern is: **don't trust LLM output; gate it, measure after, revert on failure.**

### 3. Arbitrary-code-execution on LLM output needs real isolation

`execute_code()` (line 185) writes the model's output to
`temp_execute.py` and runs it with `subprocess.Popen([sys.executable,
temp_file])` — full interpreter, full filesystem access, full
network, for 10 seconds. The user actively asked this system to
*"create a temporary e-mail. use that e-mail and find out how to
send sms"* and *"create an e-mail account. remember the username
and password"* — tasks that request real-world side effects.

FractalTrainer never executes LLM-generated code directly. It only
reads back hparams.yaml changes and re-runs the deterministic training
probe. That's a much smaller trust surface.

## Systemic problems observed (beyond the three lessons)

- **Multiple silent rewrites of `gpt_grok_ai.py` itself.** The error
  cache in `knowledge_base_excerpt.json` references
  `File "/home/vegar/AI_gpt_grok/gpt_grok_ai.py", line 166, in target` /
  `exec(code, exec_globals)` / `NameError: __build_class__ not found`
  — that's the classic `exec()` bug when `exec_globals` doesn't
  include `__builtins__`. The current `gpt_grok_ai.py` uses
  `subprocess` at line ~191, not `exec()`, so the cached errors are
  from an **earlier version of the same file**. The error messages
  in `ai_memory` and `ai_log` haven't been cleaned after refactoring.

- **Incomplete `requirements.txt`.** Only lists
  `requests, urllib3, certifi, charset-normalizer, idna`. Missing
  `openai, pyyaml, pygame, opencv-python` — all imported by the code.
  Reproducing the environment from `requirements.txt` alone won't work.

- **`Hus.py` is truncated** at line 98: last line is
  `self.screen.main` (should be `self.screen.mainloop()`). The file
  was either an LLM response cut off on token budget or the write
  was interrupted.

- **`tasks/` contains zero-byte outputs** for some tasks (e.g.
  `task_add_a_drag_and_drop*.py`, the first `task_create_flappybirds*.py`)
  — the loop saved an empty file when it thought it succeeded but
  hadn't actually produced code.

- **The `is_doom_replica` substring check** is a cautionary example
  of what "LLM-tested output" turns into without a principled metric.
  It would also pass for a code comment that happens to mention all
  five words.

## Security note (for future reference)

The source folder at `/home/vegar/AI_gpt_grok/` contained, as of
2025-03-12:

1. **`config.yaml`** — four live API keys in plaintext: OpenAI
   (`sk-h1k...`), xAI/Grok (`xai-vv...`), Anthropic Claude
   (`sk-ant-api03-1n5JV...`), Google Custom Search (`AIzaSy...`).
   **These keys should be rotated** if the folder has been reachable
   from anywhere external. Replaced here by `config.template.yaml`
   (schema only).

2. **`love_message.txt`** and `send.py` (byte-identical copies) —
   contained personal email credentials for multiple accounts plus a
   third party's full name and a love letter dated 2025-03-04.
   **Not copied to this archive.**

3. **`knowledge_base.json`** caches queries that include local
   filesystem paths (`/home/vegar/AI_gpt_grok/...`) — preserved here
   because the paths are already public via the training log but
   worth noting as a principle: **don't let debug caches egress.**

The take-away for FractalTrainer: **secrets live in environment
variables, not in config files.** `scripts/run_closed_loop.py` reads
`ANTHROPIC_API_KEY` from the environment, never from disk.

## Why this is saved rather than deleted

The code itself is superseded. But the **failure-mode patterns** are
the most important artifact: they are specific, concrete examples of
what multi-agent code-rewrite loops look like when they go wrong, and
they are the experimental evidence behind every design choice in
FractalTrainer v3. Retaining this archive lets future-you or a
reviewer answer "why did you pick THAT specific triage rule / THAT
backup-and-validate pattern / THAT scope gate?" by pointing at a real
precursor that failed in exactly the way the new design prevents.

It's the question. FractalTrainer is the answer.
