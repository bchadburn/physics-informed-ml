# Physics-Informed ML Portfolio

Four experiments applying physics-informed machine learning to industrial systems — pumps, heat transfer, compressors. Built as an interview portfolio targeting ML Engineer roles at industrial AI companies.

Each experiment is self-contained: its own data generator, model, training script, evaluation script, and tests.

---

## Experiments

### Exp 1 — PINN Surrogate with Uncertainty Quantification
`scripts/train_pump_surrogate.py`

Trains a physics-informed neural network ensemble on synthetic centrifugal pump data. Outputs calibrated uncertainty estimates (epistemic + aleatoric) with conformal prediction intervals.

- Physics residual: affinity laws (H ∝ N², Q ∝ N) as a loss term
- Ensemble uncertainty: variance across N independently trained members
- Conformal calibration: distribution-free 90% prediction intervals
- **Result:** Test RMSE ~1.5 m, 90% conformal coverage achieved

```bash
uv run python scripts/train_pump_surrogate.py
```

---

### Exp 2 — Fourier Neural Operator for Heat Transfer
`experiments/exp2_fno_heat_transfer/`

Trains FNO2d and UNet2d on 2D Darcy flow (steady-state heat equation with spatially varying conductivity κ). Tests resolution invariance and out-of-distribution generalization.

- FNO2d operates in spectral space — same learned modes generalize to any grid size
- UNet2d baseline uses fixed-scale pooling — not resolution-invariant
- OOD test: train on κ ≤ 12, evaluate on κ ≤ 24

| Condition | FNO2d Rel-L2 | UNet2d Rel-L2 |
|-----------|-------------|---------------|
| In-distribution (64×64) | 0.055 | 0.108 |
| OOD (κ ≤ 24) | 0.408 | — |
| Zero-shot 128×128 | 0.046 | — |

```bash
uv run python experiments/exp2_fno_heat_transfer/train.py
uv run python experiments/exp2_fno_heat_transfer/evaluate.py
```

---

### Exp 3 — Bayesian Residual Compressor Digital Twin
`experiments/exp3_bayesian_compressor/`

A Bayesian linear model tracking the residual between vendor efficiency curve predictions and measured compressor efficiency. Updates sequentially as new field measurements arrive.

- Conjugate Bayesian regression: closed-form posterior, no MCMC
- Rank-1 sequential update: O(d²) per new observation
- Predictive uncertainty grows in data-sparse regions
- Fouling drift: efficiency degrades 0.2% per month

| Model | RMSE (η) | Notes |
|-------|---------|-------|
| Vendor curve only | 0.0237 | Physics baseline |
| Bayesian residual | 0.0086 | Beats vendor from day 10, ECE=0.19 |

```bash
uv run python experiments/exp3_bayesian_compressor/train.py
uv run python experiments/exp3_bayesian_compressor/evaluate.py
```

---

### Exp 4 — Surrogate-in-the-Loop Optimizer (Capstone)
`experiments/exp4_surrogate_optimizer/`

Uses the Exp 1 PINN ensemble as a differentiable proxy inside a gradient-based optimizer. Minimizes pumping power subject to a minimum head constraint, comparing three strategies.

- Decision variables `(flow_rate, speed)` have `requires_grad=True`
- Gradients flow through the neural network to inputs via autograd
- Constraint handled via augmented Lagrangian: `L = P + ρ × max(0, H_min − H)²`
- Multi-start: runs N independent gradient descents, returns best feasible result

| Method | Power | Gap vs grid | Calls |
|--------|-------|-------------|-------|
| Grid search (reference) | 0.2015 | — | 10,000 |
| Random search | 0.2036 | +1% | 10,000 |
| Gradient (1 start) | 0.8706 | +332% | 501 |
| Gradient (10 starts) | 0.2012 | −0.1% | 5,010 |

Single-start gradient converges to a local minimum (expected on non-convex landscape). Multi-start recovers near-optimal solutions at 2× the call budget of random search.

```bash
uv run python experiments/exp4_surrogate_optimizer/evaluate.py
```

---

## Structure

```
configs/           Hydra configs for each experiment
core/              Shared metrics and benchmark utilities
experiments/
  exp2_fno_heat_transfer/
  exp3_bayesian_compressor/
  exp4_surrogate_optimizer/
scripts/           Top-level training scripts (Exp 1)
src/
  physics_models/  Pump physics, data generators
  surrogates/      PINN, ensemble, conformal calibration
tests/             75 tests covering all experiments
```

## Setup

```bash
uv sync
uv run pytest tests/
```

Requires Python 3.12, CUDA optional (falls back to CPU).
