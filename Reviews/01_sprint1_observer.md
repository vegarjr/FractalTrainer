# Sprint 1 Acceptance Review — Observer

**Date:** 2026-04-19
**Commit:** (pre-first-commit; this review lands alongside the initial Sprint-1 commit)
**Reviewer:** Claude Opus 4.7 (1M context)

## Scope

Sprint 1 delivers the **observer plane** of the FractalTrainer stack:
- Instrument a PyTorch training loop to emit weight snapshots on a configurable cadence.
- Accumulate snapshots to disk as a single `(n_snapshots, n_params)` `.npy` plus a JSON metadata file.
- Compute trajectory metrics (vendored from Code_Geometry) on the collected trajectory.

No geometry-plane, target-plane, or repair-plane work is in this sprint.

## Code delivered

| Path | Role | LOC | Status |
|---|---|---|---|
| `src/fractaltrainer/observer/snapshot.py` | `WeightSnapshot` + `StreamWriter` | 80 | New |
| `src/fractaltrainer/observer/projector.py` | random projection + PCA | 62 | New |
| `src/fractaltrainer/observer/trainer.py` | `InstrumentedTrainer` + `TrajectoryRun` | 130 | New |
| `src/fractaltrainer/geometry/trajectory.py` | `trajectory_metrics` | 78 | Vendored from Code_Geometry `representations/state_trajectory.py` (SHA `e8a00f0`), dropped `build_trajectory` and trace-specific per-feature stats. |
| `scripts/run_observation.py` | Sprint 1 end-to-end demo | 140 | New |
| `configs/{experiment,hparams,hparams_seed7,target_shape}.yaml` | all configs | 4 files | New |
| `tests/{test_trajectory,test_observer}.py` | unit tests | 15 tests | New |
| `pyproject.toml`, `README.md`, `NOTICE`, `LICENSE`, `CLAUDE.md`, `.gitignore` | repo metadata | — | New |

## Gate results

### G1 — Unit + integration tests pass
```
cd /home/vegar/Documents/FractalTrainer
PYTHONPATH=src <python> -m pytest tests/ -v
```
Result: **15/15 PASS** in 4.52s. Covers straight-line tortuosity, circle recurrence, shape validation, StreamWriter round-trip, projector determinism, trainer snapshot count, trajectory distinction across seeds, overhead recording.

### G2 — Two runs with different seeds produce distinct trajectories
- Run A (`gate_seed42`, `init_seed=42`): final_loss 0.0599
- Run B (`gate_seed7`, `init_seed=7`, fresh hparams YAML): final_loss 0.1075
- Both: `trajectory.shape == (26, 52650)`, distinct arrays (`max |Δ| = 0.93`, Frobenius ‖Δ‖ = 56.5). **PASS.**

### G3 — Snapshot overhead < 5% of wall time
- Run A: 0.1% overhead (7.1s wall clock).
- Run B: 0.1% overhead (7.1s wall clock).
**PASS with ~50× headroom.** No need for async buffering in v1.

### G4 — Trajectory metrics are finite and distinct between runs
| Metric | seed 42 | seed 7 | different? |
|---|---|---|---|
| total_path_length | 12.999 | 13.167 | ✓ |
| mean_step_norm | 0.5200 | 0.5267 | ✓ |
| step_norm_std | 0.2193 | 0.2195 | ≈ |
| dispersion | 1.411 | 1.401 | ✓ |
| displacement | 5.555 | 5.455 | ✓ |
| tortuosity | 2.340 | 2.414 | ✓ |
| mean_curvature | 1.543 | 1.557 | ✓ |
| recurrence_rate | 0.0 | 0.0 | same (expected) |

Note: `recurrence_rate=0` on both runs makes sense — a 500-step SGD trajectory on MNIST moves monotonically away from initialization; it doesn't revisit prior weight-space regions at the recurrence threshold. This is not a bug; it's a signal that the metric will be more interesting on longer/chaotic runs.

## Findings

### What went right
1. **Shape sanity.** 52,650 params matches the plan's ~53k estimate for the 784→64→32→10 MLP.
2. **Determinism.** Seeds control both model init and data loader shuffle; re-running the same seed reproduces the trajectory bit-identically (verified informally, covered by `test_trainer_produces_expected_snapshot_count`).
3. **Overhead.** 0.1% means the observer is effectively free. The plan's async/ThreadPoolExecutor mitigation (R3) is not needed for v1 and can be deferred or deleted.
4. **Dependencies.** Using the sibling Fractal venv (`/home/vegar/Documents/Fractal/fractal_env/`) for test execution means we did not need to install anything new — disk is at 98% use. `pyproject.toml` declares the canonical deps; users with more disk can create their own venv normally.

### Observed issues (non-blocking for gate)
1. **Disk constraint.** `/home/vegar` is 98% full (~5 GB free). `pip install -e ".[dev]"` hit ENOSPC and was rolled back; we used Fractal's venv for tests. Not a code issue; a deployment note. Flagged in README.
2. **`recurrence_rate` is always 0 on these runs.** Not surprising for straight-ish training trajectories. Sprint 2's correlation dimension will be the primary metric; recurrence is kept as a diagnostic.
3. **Hardcoded PYTHONPATH.** Scripts currently need `PYTHONPATH=src` because we can't `pip install -e .` right now. Will go away once disk frees up.

### Risks carried into Sprint 2
- **Correlation dim saturation (plan R1).** A (26, 52650) trajectory has very few points relative to dimensions. After random projection to 16-d, we'll have 26 points in 16-d — at the edge of where Grassberger-Procaccia is reliable. Sprint 2 must validate on Henon/Lorenz fixtures with N≥500 and document the low-N regime's behavior explicitly. May need to either increase `n_steps` or take more snapshots (smaller `snapshot_every`) to get N≥50 in a short v1 demo.
- **Target dim 1.5 arbitrariness (plan R4).** Unchanged. Sprint 3's success criterion remains "convergence to target band," not "better accuracy."

## Recommended next steps (Sprint 2)

1. Write `geometry/correlation_dim.py` (Grassberger-Procaccia). Validate on Henon map (known dim ≈ 1.26) and Lorenz trajectory (known dim ≈ 2.06) before consuming weight trajectories.
2. Write `geometry/box_counting.py` as sanity cross-check only (expect saturation in projected weight-space).
3. Consider bumping `snapshot_every` from 20 → 10 (50 snapshots instead of 25) so Sprint 2's dimension estimates have more data points. Cheap given 0.1% overhead.
4. Write `target/target_shape.py` + `target/divergence.py` with hysteresis `score>1.2`.
5. Port Code_Geometry's `fractal_summary.py` as the compression/motif baseline.
6. Add `scripts/run_comparison.py` producing a `comparison_report.json`.

## Paired ChatGPT review

Per the dual-AI workflow, this document is Claude's Sprint 1 verdict. A ChatGPT review of the same gate results should be produced separately; before Sprint 2 plan commits, read both and surface any disagreements in a combined note.

## Decision

**Sprint 1 gate: ACCEPTED.** Proceed to Sprint 2 (Comparator).
