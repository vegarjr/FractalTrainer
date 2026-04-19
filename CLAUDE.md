# CLAUDE.md — FractalTrainer project rules

## Project purpose
See README.md. One-line: watch a training trajectory's fractal dimension and have an LLM rewrite hyperparameters to bring it back to target.

## Hard rules

1. **Never modify sibling repos.** Code reused from `/home/vegar/Documents/Fractal/`, `/home/vegar/Documents/GeometricAI/`, `/home/vegar/Documents/SelfSupervisedGeometry/`, `/home/vegar/Documents/Code_Geometry/` is vendored into this repo. Never edit files in those directories.
2. **Vendored files carry a 5-line header** stating source path, commit SHA at copy time, license, and modifications. `NOTICE` at repo root summarizes the full vendor manifest.
3. **The repair loop may only patch `configs/hparams.yaml`.** Enforced in three places (scope, schema, outcome gates). Do not widen this scope without user approval.
4. **Keep the three planes decoupled:** observer, geometry, target+divergence, repair. New observers, new geometric metrics, or new rewrite targets should plug into clean interfaces, never require cross-plane refactors.
5. **Test fixtures must include known-dimension signals:** Cantor set (dim≈0.63), Henon map (dim≈1.26), Lorenz trajectory (dim≈2.06), unit-square fill (dim≈2.0). These are regression anchors for the geometry plane.

## Dual-AI review workflow

After every experiment or test run: write results + Claude's verdict to `Reviews/<NN>_<description>.md`, push. Before planning next steps, read both Claude's and ChatGPT's reviews jointly. (This project inherits the same dual-AI discipline as the sibling repos.)

## Style

- Research code — prefer clarity over cleverness.
- Minimal comments. Only explain *why*, never *what*.
- No speculative abstractions. Three similar lines > premature generalization.
- Tests live in `tests/`; CI runs offline (mock LLM). Real LLM calls happen locally via `--llm api|cli`.

## Milestones

- Sprint 1: Observer — instrument a training loop, emit trajectory, compute metrics. Gate: two runs with different seeds produce distinct trajectories; snapshot overhead <5%.
- Sprint 2: Comparator — fractal dimension on trajectories, target comparison, divergence. Gate: Henon returns 1.26±0.1; random walk saturates as expected.
- Sprint 3: Closed loop — vendor + adapt repair_loop, wire YAML hparams as patch target. Gate: seeded-divergent run converges to target band within 5 iterations.

## What NOT to do

- Don't pip-install the sibling repos. Always vendor.
- Don't submit LLM calls in CI. Mock for tests; real calls only in scripts.
- Don't expand `allowed_files` beyond `configs/hparams.yaml` in v1.
- Don't add features beyond the v1 plan without explicit user approval.
