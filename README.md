# Physics-Informed ML for Industrial Energy Systems

Surrogate modeling and uncertainty quantification for centrifugal pump systems, combining
physics constraints with neural networks to reduce data requirements and improve reliability.

## Approach

Three modeling layers, each building on the previous:

| Component | Purpose |
|---|---|
| **PINN** (`src/surrogates/pinn.py`) | Physics-Informed Neural Network — enforces pump affinity laws (H ∝ N²) via autograd loss, outputs heteroscedastic uncertainty |
| **Ensemble** (`src/surrogates/ensemble.py`) | Ensemble of PINNs — decomposes uncertainty into epistemic (model) and aleatoric (irreducible data noise) |
| **Bayesian Digital Twin** | Calibrates priors from physics simulation, updates posteriors with real sensor readings |

The physics loss term (`loss_physics = MSE(dH/dN, 2H/N)`) constrains the network to obey the
affinity law across all operating conditions, not just training points.

## Project Structure

```
src/
├── physics_models/     # Domain physics (pump affinity laws, data generation)
├── surrogates/         # PINN + ensemble surrogate models
├── calibration/        # Bayesian calibration of digital twin
└── optimization/       # Surrogate-in-the-loop optimization
scripts/
└── train_pump_surrogate.py   # End-to-end training script
configs/                # Hyperparameter configs (YAML)
tests/                  # Unit and integration tests
```

## Quickstart

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/bchadburn/physics-informed-ml.git
cd physics-informed-ml
uv sync
uv run python scripts/train_pump_surrogate.py
```

## Run Tests

```bash
uv run pytest tests/ -v
```

## Key Dependencies

- **PyTorch + Lightning** — PINN training with autograd physics loss
- **NumPy / SciPy** — Physics simulation and Bayesian calibration
