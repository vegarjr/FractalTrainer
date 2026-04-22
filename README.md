# FractalTrainer

This repo contains two evolving layers:

**v1 — Fractal-dimension-guided hyperparameter repair loop.** A prototype
that watches the geometric shape of a neural network's training
trajectory, measures its fractal dimension, and — when the shape drifts
outside a target band — uses a Claude LLM to rewrite hyperparameters
and retry. This is the layer the "Fractal" in the repo name refers to;
correlation dimension and box-counting are genuinely applied here.

**v3 — Mixture-of-Specialists Registry (MoSR).** A routed nearest-neighbor
architecture over independently-trained classifiers with context
injection at spawn. Described in `PAPER_DRAFT.md`. Originally titled
"Mixture-of-Fractals" until Sprint 18's audit (Reviews 42–43) measured
the architecture's structural objects and found them non-fractal in
Bourke's mathematical sense (signature cloud D = 0.69 but slope variance
0.53 fails scale-invariance). The paper was renamed; the Python
package and this repo retain *FractalTrainer* for historical continuity.

## Status

v1 (observer + repair loop) and v3 (MoSR) both ship; see Reviews/ for
the full cluster. 260+ tests passing.

## Concept

Take a trained model's weight trajectory (sequence of weight snapshots during training) as a point cloud in weight-space. Project to a lower dimension. Compute its correlation dimension (Grassberger-Procaccia). If the dimension is too far from a target value, the system asks an LLM to propose new hyperparameters, validates the patch, reruns training, measures again — and accepts only if the shape moves closer to the target.

Three-plane architecture, each extensible without rewriting the others:

```
┌─ Observer ──────────────────┐   captures (step, weight_snapshot)
│  InstrumentedTrainer         │
└───────────┬──────────────────┘
            ↓
┌─ Geometry ──────────────────┐   measures the shape
│  correlation_dim            │
│  trajectory_metrics         │
└───────────┬──────────────────┘
            ↓
┌─ Target + Divergence ───────┐   compares to goal
│  TargetShape                │
│  divergence_score           │
└───────────┬──────────────────┘
            ↓   (divergence > threshold)
┌─ Repair Loop ───────────────┐   LLM proposes fix
│  GeometricContextGatherer   │
│  PromptBuilder              │
│  PatchParser                │
│  apply → probe → accept/rollback
└──────────────────────────────┘
```

## Quick start

```bash
# 1. Install
pip install -e .
pip install -e ".[dev]"  # for tests

# 2. Sprint 1 demo — observe a training run
python scripts/run_observation.py --config configs/experiment.yaml
# → results/trajectories/run_<timestamp>.npy + metrics.json

# 3. Sprint 2 demo — compare against target (after Sprint 2)
python scripts/run_comparison.py --trajectory results/trajectories/latest.npy \
    --target configs/target_shape.yaml

# 4. Sprint 3 demo — closed loop (after Sprint 3)
python scripts/run_closed_loop.py --experiment configs/experiment.yaml \
    --target configs/target_shape.yaml --max-iters 5
```

## Repo layout

```
src/fractaltrainer/
├── observer/     # instruments a training loop, emits weight snapshots
├── geometry/     # measures fractal dim + trajectory metrics
├── target/       # target shape + divergence score
└── repair/       # LLM-driven patch loop (hparams only, scoped)

configs/          # user-editable YAML (model, dataset, target, hparams)
tests/            # pytest suite (runs offline with mock LLM)
scripts/          # end-to-end demo scripts per sprint
Reviews/          # dual-AI review workflow artifacts
results/          # runtime artifacts (gitignored)
```

## Safety model

The repair loop can only edit `configs/hparams.yaml`. Three gates:

1. **Scope** — vendored `patch_parser` rejects patches to any file outside `allowed_files=["configs/hparams.yaml"]`.
2. **Schema** — `hparam_config.validate_hparams()` rejects out-of-range values.
3. **Outcome** — every patch is backed up before apply, verified by rerunning training, and rolled back if divergence does not decrease or loss explodes.

Hysteresis (`score > 1.2` to trigger) and a hard iteration cap (`max_iters=5`) prevent flapping.

## Sibling projects

FractalTrainer reuses code from sibling research repos via **vendoring** (in-tree copies with attribution), not by importing. See `NOTICE` for the full vendor manifest. Sibling repos are **never modified** by this project.

- [Fractal](https://github.com/vegarjr/Fractal) — continual-learning projection registry; stage11 repair infrastructure vendored here.
- [GeometricAI](https://github.com/vegarjr/GeometricAI) — supervised geometric-reasoning benchmark.
- [SelfSupervisedGeometry](https://github.com/vegarjr/SelfSupervisedGeometry) — self-supervised 2D→3D latents.
- [CodeGeometryLab](https://github.com/vegarjr/CodeGeometryLab) — execution-trace geometry; trajectory + fractal-summary metrics vendored here.

## Scientific disclaimer

The target fractal dimension (v1 default: 1.5) is chosen heuristically for prototype validation. Whether this dimension correlates with any meaningful property of trained models (generalization, loss plateau, mode collapse) is an open research question, not an assumption of this system. The v1 acceptance criterion is "closed loop converges to target band," not "better test accuracy."

## Future work (v1 is prototype)

- **Self-modification observer** — watch the repair loop rewrite its own code.
- **Golden-run matching** — compare full trajectory to a stored reference, not just a scalar dim.
- **Training-code rewrites** — patch the architecture/loss, not just hyperparameters.
- **Meta-controller recursion** — tune the target/tolerance itself.
- **Online intervention** — pause mid-training, patch, resume from snapshot.
- **Non-fractal geometric targets** — persistent homology, Ricci curvature.

Each of these is additive onto v1 — no rewrites required.

## License

MIT. See `LICENSE`.
