# v3 Sprint 18 Phase 2 — rename to Mixture-of-Specialists Registry

**Date:** 2026-04-22
**Reviewer:** Claude Opus 4.7 (1M context)

## Decision

Sprint 18 Phase 1 (Review 42) measured correlation dimension on the
registry's signature cloud, growth trajectory, and label lattice.
All three objects failed the pre-registered fractal-audit gate
(pass requires non-integer D in [1.2, 9.0] AND slope variance <
0.3). Per the plan's conditional, the result triggered a rename.

The user selected **"Mixture-of-Specialists Registry" (MoSR)** as
the paper-level name with minimal scope (paper + README + this
Review).

## What changed

- `PAPER_DRAFT.md` — title rewritten from
  "Mixture-of-Fractals: a routed registry of task-specialist
  experts with context-injection" to
  "**Mixture-of-Specialists Registry: a routed nearest-neighbor
  architecture with context injection at spawn**"; abstract
  opening re-phrased accordingly.
- `README.md` — top-level description rewritten to cover both
  layers: v1 (fractal-dimension-guided hyperparameter repair —
  still genuinely fractal) and v3 (MoSR — formerly
  "Mixture-of-Fractals"). The rename rationale is stated inline
  with a pointer to Reviews 42-43.

## What did NOT change

Per the user's scope choice:

- **Python package name**: still `fractaltrainer`. Renaming would
  break every import path across integration/, geometry/,
  observer/, registry/, repair/, snake/, target/ + every
  `scripts/run_*.py`. The engineering cost outweighs the naming
  consistency.
- **Git repo name**: still `FractalTrainer`. Historical continuity
  for the repo URL and existing CI.
- **Sprint 1-16 Review titles**: unchanged. They describe work
  that genuinely was fractal-flavored (v1 hparam repair via
  correlation dimension) or that used "fractal" metaphorically at
  the time.
- **Module docstrings**: left alone. Rewriting every "Fractal"
  reference in the integration module is a separate engineering
  task that the user opted out of.

## Naming reconciliation across v1 / v3

The repo now honors a clearer split:

| Layer | Work | Fractal in Bourke's sense? |
|---|---|---|
| v1 | Observer + correlation-dim repair loop | Yes — measures fractal D of training trajectories, uses it as a target band for LLM hparam repair. |
| v3 | `integration/` Sprint 17 cluster — MoSR | No — Sprint 18 audit measured D = 0.69 on the signature cloud with slope variance 0.53 (weak-pass → fail per gate). |

The v1 layer remains "fractal" (honest). The v3 layer is now
**Mixture-of-Specialists Registry** in the paper.

## The signature cloud's D = 0.69 — a salvaged finding

The audit's headline null doesn't erase a real architectural
observation: **the 60-expert signature cloud is a low-dimensional
manifold in a 1000-d ambient space** (D = 0.69 vs random baseline
D = 13.29). This is *non-random* geometry, just not *scale-
invariant* geometry.

A routed mixture-of-experts registry whose specialists cluster on
a D ≈ 0.7 manifold is worth naming and characterizing — it implies
that N specialists don't explore the softmax-output space
randomly, they concentrate on a small subset of possible outputs.
That observation belongs in the paper's §3 under architectural
characterization, not under the fractal claim.

## Paired ChatGPT review prompts

1. **On partial salvage.** D = 0.69 on the signature cloud with
   slope variance 0.53 is "not a clean fractal, but also not
   random". In an ML paper, is that a publishable architectural
   observation or a nothing-burger that should be dropped?
2. **On the split naming (v1 = fractal, v3 = MoSR).** Is it
   intellectually honest to keep "Fractal" in v1's naming while
   removing it from v3's, or does it cheapen the audit's finding
   by letting the *repo-level* brand continue to suggest the whole
   project is fractal?
3. **On minimal-touch scope.** We rewrote paper title + README
   but kept 260 tests + all module docstrings with "fractal"
   references. Is that a pragmatic compromise or a sign of
   half-commitment — and would reviewers of the paper fault the
   inconsistency?

## What ships

- `PAPER_DRAFT.md` — title + abstract rewritten.
- `README.md` — v1/v3 split articulated.
- `Reviews/43_v3_sprint18_rename.md` — this doc.

No code changes; no tests changed; 260/260 still passing.

## Closing the Sprint 17–18 arc

Sprint 17 (12 commits) built F + C + B + D + E + G + H.
Sprint 18 (2 commits so far) audited the fractal claim and
renamed the paper when the claim didn't hold. The repo is now
honest at the paper level; the v1 layer keeps its genuine
fractal framing; and the v3 layer stands on its measured merits
as a routed MoE registry with a quantified context-injection
primitive.
