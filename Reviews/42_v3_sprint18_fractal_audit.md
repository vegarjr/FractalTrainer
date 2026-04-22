# v3 Sprint 18 — Fractal audit

**Date:** 2026-04-22
**Reviewer:** Claude Opus 4.7 (1M context)

## Purpose

FractalTrainer carries "Fractal" in its name and its core vision line
promises a *fractal geometry that expands as it learns*. Paul Bourke's
introductory essay (https://paulbourke.net/fractals/fracintro/)
defines fractal objects as possessing (i) self-similarity across
scales, (ii) non-integer dimension given by D = log(S)/log(L), (iii)
iteration or recursion, (iv) infinite complexity from simple
generative rules.

After 12 commits of Sprint 17, no part of the architecture's
*structure* has been measured against these properties. The repo
has `correlation_dim` and `box_counting` (Sprint 1-2) but uses them
only on training trajectories. This sprint applies them to three
structural objects of the registry and asks: **does the current
architecture actually exhibit fractal properties, or is the name
metaphorical?**

The answer decides Phase 2:
- **pass** (non-integer D, stable slopes on ≥1 object) → add a
  fractal mechanism (IFS spawn or recursive registry).
- **weak pass / fail** → rename the project.

## Method

`scripts/run_fractal_audit.py` measures correlation dimension (via
the existing Grassberger-Procaccia implementation) on:

1. **Signature point cloud** — the N registered signatures as points
   in R^1000.
2. **Growth trajectory** — a sequence of 50 oracle signatures trained
   on distinct binary-subset MNIST tasks drawn from ~250 combinations
   up to complement.
3. **Label-set lattice** — the class-1 indicator vectors of each
   expert in R^10.

For each: dim D, R², **scale-stability** (slope variance across
small / mid / large r-bands — a single D without slope stability
is not a fractal), and a **random-baseline D** at matched (N, ambient
dim) to control for correlation-dim saturation at high ambient dim.

Verdict thresholds:
- pass: D non-integer in [1.2, 9.0] AND slope variance < 0.3
- weak pass: D non-integer, slope variance 0.3-0.6
- fail: integer D OR variance > 0.6 OR degenerate

## Results

Registry: 60 experts (20 distinct binary-subset tasks × 3 seeds,
tasks drawn from {3..7}-element subsets of {0..9} up to complement).
Growth trajectory: 50 oracle signatures (100-step oracle each) on
distinct tasks drawn from the same pool. All on MNIST with the
Sprint-17 MLP.

Full audit took 565 seconds on one CPU.

| Object | N | D | R² | slope variance | random baseline D | Classification |
|---|---|---|---|---|---|---|
| Signature cloud   | 60 | **0.694** | 0.950 | 0.53 | 13.29 | fail |
| Growth trajectory | 50 | **5.903** | 0.999 | 0.86 |  7.17 | fail |
| Label lattice     | 60 | **0.000** | 1.000 | 0.00 |  5.94 | fail (degenerate) |

**Interpretation per object:**

- **Signature cloud**. D = 0.694 is strongly non-integer —
  Cantor-dust territory (D_Cantor ≈ 0.63). Crucially, the random
  baseline at matched (N=60, ambient=1000-d) gives D = 13.29, so
  the registry's low D is NOT saturation artifact; the signatures
  live on a low-dimensional manifold in the 1000-d space. **But**
  slope variance = 0.53 (in the "weak pass" band 0.3-0.6) means
  the slope isn't stable across r-bands. A fractal has a single
  D; a mere low-D projection does not.
- **Growth trajectory**. D = 5.903 with excellent R² = 0.999
  inside the detected scaling window, but slope variance 0.86
  means the slope varies by ±0.86 across r-bands — the object is
  *not* scale-invariant.
- **Label lattice**. D = 0 indicates degeneracy. With N = 60
  binary indicator vectors in R^10 (only 2^10 = 1024 possible
  positions), many label sets have equal or near-equal
  representations; the Grassberger-Procaccia method breaks on
  this sparse discrete structure. Using a denser embedding or a
  Jaccard-kernel metric might help, but the object is too coarse
  for correlation-dim at this N.

## Verdict

**FAIL on all three objects** per the acceptance criteria (pass
requires non-integer D in [1.2, 9.0] AND slope variance < 0.3).

Honest reading:

- The **signature cloud has real low-dimensional structure** (D ≈ 0.7
  vs random baseline D ≈ 13.3). This is a meaningful finding —
  the registry does not fill signature space randomly. It lives on
  a small, potentially-informative manifold.
- But the **structure is not scale-invariant**. Slope variance 0.53
  puts it in the "weak pass" band by the plan's bookkeeping, which
  rolls up to `overall = fail` because we don't have a clean pass
  on any object.
- The **trajectory and lattice objects give no evidence of fractal
  structure** at all.

The architecture has a *non-random* signature geometry but not a
*fractal* one in Bourke's mathematical sense.

## Phase 2 direction

Per the pre-registered gate ("weak pass / fail → rename"): the
project should be renamed.

Concrete actions for Phase 2 (Review 43):
1. Rewrite `PAPER_DRAFT.md` title and abstract. Proposed names:
   - "Mixture-of-Specialists Registry with Context-Injection"
   - "Routed MoE Registry: Signature-Space Specialist Dispatch"
   - "Context-Augmented Specialist Dispatch"
   Let the user pick.
2. Update `README.md` and top-of-file docstrings across the
   `integration/` module.
3. **Do not rename the Python package or git repo** — breaking
   existing import paths is a separate engineering cost that
   outweighs the naming cleanliness.
4. Document the audit + rename decision in Review 43.
5. Keep `fractal_analysis.py` — it's a genuine contribution
   regardless of whether the architecture it analyzes is called
   "fractal" or "routed MoE". The result *"60 experts cluster on
   a D ≈ 0.7 sub-manifold"* is a real architectural fact worth
   reporting.

## What ships

- `src/fractaltrainer/integration/fractal_analysis.py` — audit helpers
  (signature cloud, growth trajectory, label lattice) + scale-stability
  computation + `classify_verdict`
- `scripts/run_fractal_audit.py` — driver (smoke + full modes)
- `tests/test_fractal_analysis.py` — 11 unit tests (Cantor ≈0.63,
  Henon ≈1.26, unit square ≈2, iid gaussian saturates, + shape /
  error paths)
- `results/fractal_audit.json` + `results/fractal_audit_scaling.png`
- `Reviews/42_v3_sprint18_fractal_audit.md` — this doc

## Paired ChatGPT review prompts

1. **On the "single-D is not enough" framing.** I'm requiring slope
   stability across three r-bands before calling something fractal.
   Is that overkill (Grassberger-Procaccia's D is already a fit over
   a detected scaling window) or essential (the detected window's
   slope could still vary within itself)? What's the standard
   acceptance threshold in the fractal-measurement literature?
2. **On N=60 being small.** Correlation dimension is known to be
   unreliable below N≈10²–10³. The audit's N=60 signatures are
   borderline. Does the random-baseline comparison (reporting the
   ratio of measured D to baseline D at matched N) provide enough
   guard-rail, or is the result too noisy to claim anything?
3. **On the pass/fail gate for renaming.** The plan treats
   "weak_pass" (variance 0.3-0.6) as equivalent to fail, on the
   grounds that a paper shouldn't carry a hedged fractal claim.
   Is that the right default, or should weak-pass instead trigger
   one of the implementation directions (IFS / recursive) as a
   way to *create* fractal structure where the audit found a hint?
