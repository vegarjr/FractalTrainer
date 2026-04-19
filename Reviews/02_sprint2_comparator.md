# Sprint 2 Acceptance Review — Comparator

**Date:** 2026-04-19
**Reviewer:** Claude Opus 4.7 (1M context)

## Scope

Sprint 2 delivers the **geometry plane** and the **target + divergence plane**:

- `geometry/correlation_dim.py` — Grassberger-Procaccia on (N, D) point clouds.
- `geometry/box_counting.py` — standard box-counting, used as a sanity cross-check in low ambient dimensions.
- `geometry/fractal_summary.py` — vendored + adapted from Code_Geometry; baseline proxy metrics (compression ratio, motif entropy, self-similarity).
- `target/target_shape.py` — `TargetShape` dataclass + YAML loader with validation.
- `target/divergence.py` — `divergence_score`, `should_intervene` (hysteresis), `within_band`.
- `scripts/run_comparison.py` — end-to-end demo taking a trajectory + target → `comparison_report.json`.
- `tests/fixtures.py` — known-dimension fixtures (Cantor, Henon, Lorenz, unit square, high-d random walk).

## Gate results

### G1 — Full test suite passes
```
PYTHONPATH=src <python> -m pytest tests/ -q
```
Result: **42/42 PASS** in 4.37s (was 15 in Sprint 1; +27 new tests).

### G2 — Regression anchors (known fractal fixtures)
| Fixture | Theoretical dim | Method | Measured | Threshold | Status |
|---|---|---|---|---|---|
| Cantor set (N=3000, L=10) | log(2)/log(3) ≈ 0.631 | box-counting | inside [0.45, 0.80] | bounds | PASS |
| Henon attractor (N=2000) | 1.26 | correlation dim | inside [1.1, 1.4] | bounds | PASS |
| Lorenz trajectory (N=3000) | 2.05 | correlation dim | inside [1.7, 2.3] | bounds | PASS |
| Unit square fill (N=2000) | 2.0 | box-counting | inside [1.7, 2.1] | bounds | PASS |
| Unit square fill (N=2000) | 2.0 | correlation dim | inside [1.5, 2.2] | bounds* | PASS |
| 16-D random walk (N=500) | saturates, target-diverges | correlation dim | > 1.8 | edge of target band | PASS |

\* Correlation dim on N=2000 uniform 2D points commonly lands around 1.7–1.8 due to boundary effects. This is a documented finite-N behavior, not a bug. Box-counting in the same test gives a cleaner ≈2.0.

### G3 — Target + divergence logic
All 9 `test_target.py` tests pass:
- Center (dim == target_dim) gives score 0.
- Band edges (target ± tolerance) give score 1.
- Outside the band gives score > 1.
- NaN dim gives inf score (correctly triggers intervention).
- Hysteresis: score = 1.1 (inside band edge) → no intervention; score = 1.3 → intervention.
- `load_target` rejects `tolerance <= 0`, unknown `method`, unknown projection method.

### G4 — End-to-end comparator on a real trajectory
```
PYTHONPATH=src ... scripts/run_comparison.py \
    --trajectory results/trajectories/gate_seed42_trajectory.npy
```
Output:
- trajectory shape `(26, 52650)` → random projection to `(26, 16)`.
- Primary correlation dim: **0.37** (r² = 0.996).
- Sanity box-counting 3d: **0.16**.
- Target 1.5 ± 0.3 → divergence score **3.78**, within_band=False, should_intervene=**True**.
- Report saved to `results/trajectories/gate_seed42_trajectory.comparison_report.json`.

The comparator **correctly flags** the seeded-divergent training run (SGD at lr=0.1, 500 steps on MNIST) as geometrically far from the target fractal dim. Primary and sanity methods agree on the verdict (both low-dim, both outside band).

## Findings

### What held up
1. **Henon and Lorenz fixtures return their theoretical dimensions** within reasonable tolerance. These are the two canonical deterministic-chaos reference systems for correlation dim validation; passing them both is a strong sanity gate.
2. **Scaling-range detector** (`_auto_scaling_range`) initially preferred short windows (narrow locally-optimal fits). Fixed by changing the scoring rule to "among windows within 2× of the best RSS-per-point, pick the longest." This is more robust to noise in the log-log curve.
3. **Divergence logic is clean.** NaN handling, hysteresis, and band-edge semantics all work as expected.
4. **End-to-end flow works** — the observer from Sprint 1 and the comparator from Sprint 2 compose without any glue changes.

### Honest caveats
1. **N=26 snapshots is at the edge of GP's reliability.** Correlation dim with only 26 points returns a noisy estimate — we got 0.37 here. The `min_snapshots=20` threshold allows the computation, but downstream interpretation should treat dim values from such short trajectories as directional indicators, not precise measurements. This will become more reliable in Sprint 3's closed-loop demo where we can raise `snapshot_every` to get more points cheaply (Sprint 1 showed the observer overhead is 0.1%).
2. **Unit-square finite-N dimension** returns 1.7–1.8 for correlation dim on N=2000 uniform 2D points — below the textbook "exactly 2." This is boundary-effect bias, not an implementation bug. Box-counting gives a cleaner 2.0, which is why it's retained as a sanity cross-check.
3. **High-D walk is only "diverged from target," not "saturated at 16."** A 500-step random walk in 16-D is a trajectory (not a point cloud), so its correlation dim reports ~1.9, not 16. For the system's purpose this is sufficient: 1.9 is outside the [1.2, 1.8] target band → divergence detected. Precise saturation was never the goal.
4. **The vendored `fractal_summary` is adapted, not identical.** I retargeted inputs from trace-event records to numeric (N, D) arrays via per-dim binning. The metric concept is preserved (compression, motif, self-similarity) but this is substantially modified, not a verbatim vendor. The header on the file documents exactly what changed.

### Observed non-issues
- All existing Sprint 1 tests (15) still pass unmodified.
- No changes required to `observer/`.
- The plan's pyproject.toml already declared `scikit-learn` (needed for GaussianRandomProjection); no dep change.

### Risks carried into Sprint 3
- **N<50 regime.** If the closed loop runs 500 training steps with snapshot_every=20, we get 25 snapshots. GP at that N is noisy. Options: bump `snapshot_every` to 5-10 for the closed-loop demo; or accept noisy measurements and rely on the repair loop to converge over multiple iterations. Recommendation: bump to 10, documented in `configs/experiment.yaml` comments.
- **LLM sandboxing.** Sprint 3 needs a mock LLM in CI and a real LLM locally. Fractal's `llm_client.py` (to be vendored) handles both via a client-protocol interface; no new risk here beyond what was planned.
- **scipy dep.** `fractal_summary` doesn't need scipy (used zlib + Counter only). Correlation dim uses numpy polyfit. No scipy required in FractalTrainer — confirmed current pyproject is sufficient.

## Decision

**Sprint 2 gate: ACCEPTED.** Proceed to Sprint 3 (Closed loop).

Before Sprint 3 starts:
- Bump `snapshot_every` in `configs/experiment.yaml` from 20 → 10 so the closed loop has ~50 points per retraining round instead of 25.
- Verify `anthropic` Python package is available (it's declared in `pyproject.toml`; confirm it's importable from Fractal's venv or whatever venv we run Sprint 3 against).

## Paired ChatGPT review

Per the dual-AI workflow, this document is Claude's Sprint 2 verdict. ChatGPT's parallel review should be produced separately; before Sprint 3 plan commits, read both and surface any disagreements.
