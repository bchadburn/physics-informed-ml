# Week 1: Bayesian Pump Curve Surrogate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a physics-informed neural network ensemble that predicts centrifugal pump head with calibrated uncertainty, trained on synthetic field data with simulated degradation.

**Architecture:** A `PumpPhysics` class encodes vendor curve + affinity laws. A synthetic data generator produces 500 labeled measurements with realistic degradation noise. A PINN (PyTorch Lightning) adds physics residual loss via autograd. A `PINNEnsemble` wraps 10 PINNs to decompose epistemic vs aleatoric uncertainty. Training is configured via Hydra and tracked in MLflow.

**Tech Stack:** Python 3.11, PyTorch 2.2, PyTorch Lightning, Hydra-core, Pydantic v2, MLflow, uv

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pyproject.toml` | Create | uv project, all dependencies |
| `.gitignore` | Create | Python, Jupyter, MLflow, model artifacts |
| `configs/base.yaml` | Create | Shared Hydra defaults |
| `configs/pump_surrogate.yaml` | Create | Experiment-specific overrides |
| `src/__init__.py` | Create | Package root |
| `src/physics_models/__init__.py` | Create | Subpackage |
| `src/physics_models/pump.py` | Create | `PumpParameters`, `PumpPhysics` |
| `src/physics_models/data_generator.py` | Create | `generate_pump_field_data()` |
| `src/surrogates/__init__.py` | Create | Subpackage |
| `src/surrogates/pinn.py` | Create | `PINN` Lightning module |
| `src/surrogates/ensemble.py` | Create | `PINNEnsemble` |
| `scripts/train_pump_surrogate.py` | Create | Hydra training entry point |
| `tests/__init__.py` | Create | Test package |
| `tests/test_pump_physics.py` | Create | Physics model unit tests |
| `tests/test_pinn.py` | Create | Surrogate unit tests |

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/__init__.py`, `src/physics_models/__init__.py`, `src/surrogates/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/.gitkeep`, `models/.gitkeep`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "physics-informed-ml"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "pytorch-lightning>=2.2",
    "hydra-core>=1.3",
    "pydantic>=2.0",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "mlflow>=2.11",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "scipy>=1.12",
    "numpy>=1.26",
    "pandas>=2.2",
    "h5py>=3.10",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.venv/
dist/
.ipynb_checkpoints/
*.ipynb_checkpoints
mlruns/
mlartifacts/
models/
data/*.csv
data/*.h5
data/*.parquet
*.pt
*.ckpt
.env
```

- [ ] **Step 3: Create package `__init__.py` files and placeholder dirs**

```bash
mkdir -p src/physics_models src/surrogates src/calibration src/optimization src/api
mkdir -p tests data models scripts experiments configs
touch src/__init__.py src/physics_models/__init__.py src/surrogates/__init__.py
touch src/calibration/__init__.py src/optimization/__init__.py src/api/__init__.py
touch tests/__init__.py data/.gitkeep models/.gitkeep
```

- [ ] **Step 4: Install dependencies**

```bash
uv venv && uv pip install -e ".[dev]"
```

Expected: environment created, all packages installed without error.

- [ ] **Step 5: Verify import works**

```bash
uv run python -c "import torch; import lightning; import hydra; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/ data/.gitkeep models/.gitkeep scripts/ experiments/ configs/
git commit -m "Scaffold project structure and dependencies"
```

---

## Task 2: Pump physics model

**Files:**
- Create: `src/physics_models/pump.py`
- Test: `tests/test_pump_physics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pump_physics.py
import numpy as np
import pytest
from src.physics_models.pump import PumpParameters, PumpPhysics


def test_head_at_design_point():
    """At design flow and speed, predicted head should match design head."""
    params = PumpParameters(
        design_flow=0.05,       # m³/s
        design_head=30.0,       # m
        design_speed=1450.0,    # rpm
        design_efficiency=0.75,
    )
    physics = PumpPhysics(params)
    h = physics.head(flow=0.05, speed=1450.0)
    assert abs(h - 30.0) < 1.0, f"Expected ~30m, got {h:.2f}m"


def test_affinity_law_head_scales_with_speed_squared():
    """Doubling speed should quadruple head (affinity law)."""
    params = PumpParameters(
        design_flow=0.05,
        design_head=30.0,
        design_speed=1450.0,
        design_efficiency=0.75,
    )
    physics = PumpPhysics(params)
    h1 = physics.head(flow=0.03, speed=1450.0)
    h2 = physics.head(flow=0.06, speed=2900.0)  # 2x speed, 2x flow (affinity)
    ratio = h2 / h1
    assert abs(ratio - 4.0) < 0.5, f"Expected head ratio ~4, got {ratio:.2f}"


def test_efficiency_peak_near_design_flow():
    """Efficiency should be highest near design flow."""
    params = PumpParameters(
        design_flow=0.05,
        design_head=30.0,
        design_speed=1450.0,
        design_efficiency=0.75,
    )
    physics = PumpPhysics(params)
    eta_design = physics.efficiency(flow=0.05, speed=1450.0)
    eta_low = physics.efficiency(flow=0.01, speed=1450.0)
    eta_high = physics.efficiency(flow=0.09, speed=1450.0)
    assert eta_design > eta_low
    assert eta_design > eta_high


def test_power_positive():
    """Hydraulic power must always be positive for positive flow."""
    params = PumpParameters(
        design_flow=0.05,
        design_head=30.0,
        design_speed=1450.0,
        design_efficiency=0.75,
    )
    physics = PumpPhysics(params)
    flows = np.linspace(0.01, 0.09, 20)
    for q in flows:
        p = physics.power(flow=q, speed=1450.0)
        assert p > 0, f"Power must be positive, got {p:.2f} at flow={q:.3f}"


def test_invalid_flow_raises():
    """Negative flow should raise ValueError."""
    params = PumpParameters(
        design_flow=0.05,
        design_head=30.0,
        design_speed=1450.0,
        design_efficiency=0.75,
    )
    physics = PumpPhysics(params)
    with pytest.raises(ValueError, match="flow"):
        physics.head(flow=-0.01, speed=1450.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_pump_physics.py -v
```

Expected: `ImportError: cannot import name 'PumpParameters'`

- [ ] **Step 3: Implement `src/physics_models/pump.py`**

```python
"""Centrifugal pump physics model using affinity laws and parabolic curves.

Reference: Grundfos Pump Handbook, Chapter 1 (pump curve fundamentals).
Affinity laws: Q ∝ N, H ∝ N², P ∝ N³
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, field_validator


class PumpParameters(BaseModel):
    """Vendor-supplied design point and curve shape for a centrifugal pump."""

    design_flow: float        # m³/s at best efficiency point (BEP)
    design_head: float        # m at BEP
    design_speed: float       # rpm at BEP
    design_efficiency: float  # dimensionless [0, 1] at BEP

    @field_validator("design_flow", "design_head", "design_speed")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Must be positive, got {v}")
        return v

    @field_validator("design_efficiency")
    @classmethod
    def efficiency_in_range(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError(f"Efficiency must be in (0, 1), got {v}")
        return v


class PumpPhysics:
    """Physics model for a centrifugal pump.

    Head-flow curve:    H(Q) = a - b*Q²   (parabolic, fitted to design point)
    Efficiency curve:   η(Q) = c*Q - d*Q² (parabolic, peak at BEP)
    Affinity laws scale head and flow with speed ratio.
    """

    def __init__(self, params: PumpParameters) -> None:
        self.params = params
        # Fit parabolic coefficients so the curve passes through:
        #   H(0) = shutoff head = 1.33 * design_head (typical)
        #   H(design_flow) = design_head
        Q0 = params.design_flow
        H0 = params.design_head
        self._a = 1.33 * H0                    # shutoff head
        self._b = (self._a - H0) / (Q0 ** 2)  # curvature

        # Efficiency curve: η(Q) = c*Q - d*Q²
        # Constraints: η(Q0) = design_efficiency, dη/dQ|Q0 = 0 (peak at BEP)
        # From dη/dQ = c - 2d*Q0 = 0 → c = 2*d*Q0
        # From η(Q0) = c*Q0 - d*Q0² = d*Q0² → d = design_efficiency / Q0²
        self._d = params.design_efficiency / (Q0 ** 2)
        self._c = 2 * self._d * Q0

    def _scale_to_speed(self, flow: float, speed: float) -> tuple[float, float]:
        """Apply affinity laws: scale flow and head reference to given speed."""
        ratio = speed / self.params.design_speed
        q_ref = flow / ratio          # equivalent flow at design speed
        return q_ref, ratio

    def head(self, flow: float, speed: float) -> float:
        """Predict head [m] at given flow [m³/s] and speed [rpm]."""
        if flow < 0:
            raise ValueError(f"flow must be >= 0, got {flow}")
        q_ref, ratio = self._scale_to_speed(flow, speed)
        h_ref = self._a - self._b * q_ref ** 2
        return h_ref * ratio ** 2

    def efficiency(self, flow: float, speed: float) -> float:
        """Predict isentropic efficiency [0, 1] at given flow and speed."""
        if flow < 0:
            raise ValueError(f"flow must be >= 0, got {flow}")
        q_ref, _ = self._scale_to_speed(flow, speed)
        eta = self._c * q_ref - self._d * q_ref ** 2
        return float(np.clip(eta, 0.0, 1.0))

    def power(self, flow: float, speed: float, rho: float = 1000.0) -> float:
        """Predict shaft power [W] at given flow and speed.

        P = rho * g * Q * H / eta
        rho: fluid density [kg/m³], default water at 20°C
        """
        if flow < 0:
            raise ValueError(f"flow must be >= 0, got {flow}")
        g = 9.81
        h = self.head(flow, speed)
        eta = self.efficiency(flow, speed)
        eta = max(eta, 1e-6)  # avoid division by zero at zero flow
        return rho * g * flow * h / eta
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_pump_physics.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/physics_models/pump.py tests/test_pump_physics.py
git commit -m "Add PumpPhysics model with affinity laws and parabolic curves"
```

---

## Task 3: Synthetic field data generator

**Files:**
- Create: `src/physics_models/data_generator.py`
- Test: `tests/test_pump_physics.py` (extend)

- [ ] **Step 1: Add failing tests (append to `tests/test_pump_physics.py`)**

```python
# Append to tests/test_pump_physics.py

from src.physics_models.data_generator import generate_pump_field_data


def test_data_generator_shape():
    """Generated DataFrame has correct columns and row count."""
    df = generate_pump_field_data(n_samples=500, seed=42)
    expected_cols = {"timestamp", "flow_rate", "head", "speed", "operating_hours"}
    assert expected_cols.issubset(df.columns), f"Missing columns: {expected_cols - set(df.columns)}"
    assert len(df) == 500


def test_data_generator_degradation():
    """Head values late in operation should average lower than early operation."""
    df = generate_pump_field_data(n_samples=500, seed=42)
    early = df[df["operating_hours"] < 1000]["head"].mean()
    late = df[df["operating_hours"] > 7000]["head"].mean()
    assert late < early, "Degraded pump should have lower average head"


def test_data_generator_reproducible():
    """Same seed produces identical DataFrames."""
    df1 = generate_pump_field_data(n_samples=100, seed=0)
    df2 = generate_pump_field_data(n_samples=100, seed=0)
    import pandas as pd
    pd.testing.assert_frame_equal(df1, df2)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_pump_physics.py::test_data_generator_shape -v
```

Expected: `ImportError: cannot import name 'generate_pump_field_data'`

- [ ] **Step 3: Implement `src/physics_models/data_generator.py`**

```python
"""Synthetic pump field data generator with simulated degradation.

Simulates 500 operating measurements spanning ~1 year of pump operation.
Degradation model:
  - Impeller wear: head coefficient drops 5–15% linearly over 8760 hours
  - Seal leakage: flow measurement offset +2–8% (random per unit)
  - Measurement noise: ±3% on flow, ±5% on head (Gaussian)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.physics_models.pump import PumpParameters, PumpPhysics

# Default design-point pump (mid-range industrial centrifugal)
_DEFAULT_PARAMS = PumpParameters(
    design_flow=0.05,       # 50 L/s
    design_head=30.0,       # 30 m
    design_speed=1450.0,    # rpm
    design_efficiency=0.75,
)


def generate_pump_field_data(
    n_samples: int = 500,
    seed: int = 42,
    params: PumpParameters = _DEFAULT_PARAMS,
    max_hours: float = 8760.0,   # one year of operation
) -> pd.DataFrame:
    """Generate synthetic pump operating measurements with degradation.

    Args:
        n_samples: Number of measurement rows to generate.
        seed: Random seed for reproducibility.
        params: Pump design parameters.
        max_hours: Total operating hours span.

    Returns:
        DataFrame with columns: timestamp, flow_rate, head, speed,
        operating_hours.
    """
    rng = np.random.default_rng(seed)
    physics = PumpPhysics(params)

    # Operating hours distributed unevenly (clustered around business hours)
    operating_hours = np.sort(rng.uniform(0, max_hours, n_samples))

    # Degradation factors: progress from 0 (new) to 1 (end of life)
    degradation_progress = operating_hours / max_hours

    # Impeller wear: reduces head by 5–15% over lifetime
    wear_fraction = rng.uniform(0.05, 0.15)
    head_wear_factor = 1.0 - wear_fraction * degradation_progress

    # Seal leakage: constant flow measurement offset per pump instance
    seal_leak_fraction = rng.uniform(0.02, 0.08)

    # Speed: varies around nominal ±5% (VFD or load variation)
    speed = rng.uniform(
        params.design_speed * 0.85,
        params.design_speed * 1.05,
        n_samples,
    )

    # True flow: randomly sampled operating points across the curve
    q_true = rng.uniform(
        params.design_flow * 0.3,
        params.design_flow * 1.4,
        n_samples,
    )

    # True head from physics model + wear degradation
    h_true = np.array([
        physics.head(q, s) * hw
        for q, s, hw in zip(q_true, speed, head_wear_factor)
    ])

    # Measurements: add sensor noise
    flow_noise = rng.normal(0, 0.03, n_samples)   # ±3% relative
    head_noise = rng.normal(0, 0.05, n_samples)   # ±5% relative

    flow_measured = q_true * (1.0 + seal_leak_fraction + flow_noise)
    head_measured = h_true * (1.0 + head_noise)

    # Timestamps: 1-hour resolution from arbitrary start
    start = pd.Timestamp("2024-01-01")
    timestamps = [
        start + pd.Timedelta(hours=float(h)) for h in operating_hours
    ]

    return pd.DataFrame({
        "timestamp": timestamps,
        "flow_rate": flow_measured,
        "head": head_measured,
        "speed": speed,
        "operating_hours": operating_hours,
    })
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_pump_physics.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Generate and save the dataset**

```bash
uv run python -c "
from src.physics_models.data_generator import generate_pump_field_data
df = generate_pump_field_data(n_samples=500, seed=42)
df.to_csv('data/pump_field_measurements.csv', index=False)
print(df.describe())
print('Saved to data/pump_field_measurements.csv')
"
```

Expected: stats table showing flow_rate ~0.04–0.07, head ~10–45, operating_hours 0–8760.

- [ ] **Step 6: Commit**

```bash
git add src/physics_models/data_generator.py tests/test_pump_physics.py
git commit -m "Add synthetic pump field data generator with degradation model"
```

---

## Task 4: Hydra configs

**Files:**
- Create: `configs/base.yaml`
- Create: `configs/pump_surrogate.yaml`

- [ ] **Step 1: Create `configs/base.yaml`**

```yaml
# configs/base.yaml
defaults:
  - _self_

seed: 42

data:
  csv_path: data/pump_field_measurements.csv
  train_split: 0.6
  val_split: 0.2
  # test_split is remainder (0.2)

logging:
  mlflow_tracking_uri: mlruns
  experiment_name: pump_surrogate
```

- [ ] **Step 2: Create `configs/pump_surrogate.yaml`**

```yaml
# configs/pump_surrogate.yaml
defaults:
  - base
  - _self_

model:
  hidden_dims: [64, 64, 64]
  activation: tanh
  dropout: 0.0

training:
  lr: 1.0e-3
  batch_size: 64
  max_epochs: 300
  early_stop_patience: 30

physics:
  lambda_data: 1.0
  lambda_physics: 0.1

ensemble:
  n_members: 10
```

- [ ] **Step 3: Commit**

```bash
git add configs/
git commit -m "Add Hydra configs for pump surrogate experiment"
```

---

## Task 5: PINN Lightning module

**Files:**
- Create: `src/surrogates/pinn.py`
- Test: `tests/test_pinn.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pinn.py
import torch
import pytest
from src.surrogates.pinn import PINN, PINNConfig


def _make_batch(n: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X, y) where X = [flow, speed, hours], y = head."""
    X = torch.rand(n, 3)
    X[:, 0] *= 0.1    # flow: 0–0.1 m³/s
    X[:, 1] = X[:, 1] * 1000 + 1000   # speed: 1000–2000 rpm
    X[:, 2] *= 8760   # hours: 0–8760
    y = torch.rand(n, 1) * 40 + 5     # head: 5–45 m
    return X, y


def test_pinn_forward_shape():
    """Forward pass returns (mean, logvar) both shape [N, 1]."""
    config = PINNConfig(hidden_dims=[32, 32], activation="tanh")
    model = PINN(config)
    X, _ = _make_batch(8)
    mean, logvar = model(X)
    assert mean.shape == (8, 1), f"Expected (8,1), got {mean.shape}"
    assert logvar.shape == (8, 1), f"Expected (8,1), got {logvar.shape}"


def test_pinn_loss_has_data_and_physics_components():
    """Loss dict must contain 'loss_data' and 'loss_physics' keys."""
    from src.physics_models.pump import PumpParameters, PumpPhysics
    config = PINNConfig(hidden_dims=[32, 32], activation="tanh",
                        lambda_data=1.0, lambda_physics=0.1)
    physics = PumpPhysics(PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    ))
    model = PINN(config, physics=physics)
    X, y = _make_batch(16)
    X.requires_grad_(True)
    losses = model.compute_losses(X, y)
    assert "loss_data" in losses
    assert "loss_physics" in losses
    assert "loss_total" in losses


def test_pinn_physics_loss_shape_independent_of_batch():
    """Physics loss should be a scalar regardless of batch size."""
    from src.physics_models.pump import PumpParameters, PumpPhysics
    config = PINNConfig(hidden_dims=[32, 32], activation="tanh",
                        lambda_data=1.0, lambda_physics=0.1)
    physics = PumpPhysics(PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    ))
    model = PINN(config, physics=physics)
    for n in [8, 32, 128]:
        X, y = _make_batch(n)
        X.requires_grad_(True)
        losses = model.compute_losses(X, y)
        assert losses["loss_physics"].ndim == 0, "Physics loss must be scalar"


def test_pinn_training_step_decreases_loss():
    """After 50 gradient steps, total loss should decrease."""
    from src.physics_models.pump import PumpParameters, PumpPhysics
    config = PINNConfig(hidden_dims=[32, 32], activation="tanh",
                        lambda_data=1.0, lambda_physics=0.1, lr=1e-3)
    physics = PumpPhysics(PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    ))
    model = PINN(config, physics=physics)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X, y = _make_batch(64)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        X_grad = X.clone().requires_grad_(True)
        loss_dict = model.compute_losses(X_grad, y)
        loss_dict["loss_total"].backward()
        optimizer.step()
        losses.append(loss_dict["loss_total"].item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
    )
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_pinn.py -v
```

Expected: `ImportError: cannot import name 'PINN'`

- [ ] **Step 3: Implement `src/surrogates/pinn.py`**

```python
"""Physics-Informed Neural Network for centrifugal pump head prediction.

Architecture:
  Input:  [flow_rate, speed, operating_hours]  (normalized)
  Hidden: N layers × width units, Tanh activation
  Output: [head_mean, head_logvar]  (heteroscedastic uncertainty)

Physics loss: enforces H ∝ N² (affinity law) via autograd.
  dH/dN should equal 2*H/N at any operating point.
  loss_physics = MSE(dH/dN, 2*H/N)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import lightning as L
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.physics_models.pump import PumpPhysics


@dataclass
class PINNConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64, 64])
    activation: str = "tanh"
    dropout: float = 0.0
    lambda_data: float = 1.0
    lambda_physics: float = 0.1
    lr: float = 1e-3


class PINN(L.LightningModule):
    """Physics-Informed Neural Network with heteroscedastic uncertainty output."""

    def __init__(
        self,
        config: PINNConfig,
        physics: "PumpPhysics | None" = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.physics = physics

        # Input normalization statistics (set during fit, defaults are safe)
        self.register_buffer("x_mean", torch.zeros(3))
        self.register_buffer("x_std", torch.ones(3))

        # Build MLP
        act = nn.Tanh() if config.activation == "tanh" else nn.ReLU()
        layers: list[nn.Module] = []
        in_dim = 3
        for h in config.hidden_dims:
            layers += [nn.Linear(in_dim, h), act]
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        # Two outputs: mean and log-variance
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (head_mean, head_logvar), each shape [N, 1]."""
        x_norm = (x - self.x_mean) / (self.x_std + 1e-8)
        out = self.net(x_norm)
        mean = out[:, :1]
        logvar = out[:, 1:]
        return mean, logvar

    def compute_losses(
        self, x: torch.Tensor, y_head: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute data loss + physics residual loss.

        x columns: [flow_rate (0), speed (1), operating_hours (2)]
        y_head: true head measurements, shape [N, 1]
        """
        mean, logvar = self(x)

        # Heteroscedastic negative log-likelihood
        # NLL = 0.5 * (logvar + (y - mean)² / exp(logvar))
        loss_data = 0.5 * (logvar + (y_head - mean) ** 2 / (logvar.exp() + 1e-8))
        loss_data = loss_data.mean()

        # Physics loss: affinity law H ∝ N²
        # d(mean)/d(speed) should equal 2 * mean / speed
        if x.requires_grad:
            grad = torch.autograd.grad(
                outputs=mean.sum(),
                inputs=x,
                create_graph=True,
            )[0]
            dH_dN = grad[:, 1:2]                     # derivative w.r.t. speed col
            speed = x[:, 1:2]
            target_dH_dN = 2.0 * mean.detach() / (speed + 1e-8)
            loss_physics = nn.functional.mse_loss(dH_dN, target_dH_dN)
        else:
            loss_physics = torch.tensor(0.0, device=x.device)

        loss_total = (
            self.config.lambda_data * loss_data
            + self.config.lambda_physics * loss_physics
        )
        return {
            "loss_data": loss_data,
            "loss_physics": loss_physics,
            "loss_total": loss_total,
        }

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        x = x.requires_grad_(True)
        losses = self.compute_losses(x, y)
        self.log("train/loss_total", losses["loss_total"], prog_bar=True)
        self.log("train/loss_data", losses["loss_data"])
        self.log("train/loss_physics", losses["loss_physics"])
        return losses["loss_total"]

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        x = x.requires_grad_(True)
        losses = self.compute_losses(x, y)
        self.log("val/loss_total", losses["loss_total"], prog_bar=True)
        self.log("val/loss_physics", losses["loss_physics"])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_pinn.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surrogates/pinn.py tests/test_pinn.py
git commit -m "Add PINN with heteroscedastic output and affinity law physics loss"
```

---

## Task 6: Ensemble wrapper

**Files:**
- Create: `src/surrogates/ensemble.py`
- Test: `tests/test_pinn.py` (extend)

- [ ] **Step 1: Add failing tests (append to `tests/test_pinn.py`)**

```python
# Append to tests/test_pinn.py

from src.surrogates.ensemble import PINNEnsemble, EnsemblePrediction


def test_ensemble_predict_returns_uncertainty():
    """Ensemble prediction must return mean, epistemic_std, aleatoric_std."""
    from src.physics_models.pump import PumpParameters, PumpPhysics
    from src.surrogates.pinn import PINNConfig
    config = PINNConfig(hidden_dims=[16, 16], activation="tanh")
    physics = PumpPhysics(PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    ))
    ensemble = PINNEnsemble(config=config, n_members=3, physics=physics)
    X, _ = _make_batch(8)
    pred = ensemble.predict(X)
    assert isinstance(pred, EnsemblePrediction)
    assert pred.mean.shape == (8, 1)
    assert pred.epistemic_std.shape == (8, 1)
    assert pred.aleatoric_std.shape == (8, 1)
    assert pred.total_std.shape == (8, 1)


def test_ensemble_epistemic_uncertainty_increases_ood():
    """Epistemic uncertainty should be higher far outside training range."""
    from src.physics_models.pump import PumpParameters, PumpPhysics
    from src.surrogates.pinn import PINNConfig
    import pandas as pd
    from src.physics_models.data_generator import generate_pump_field_data
    from torch.utils.data import TensorDataset, DataLoader

    config = PINNConfig(hidden_dims=[32, 32], activation="tanh",
                        lambda_data=1.0, lambda_physics=0.0, lr=1e-3)
    physics = PumpPhysics(PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    ))
    ensemble = PINNEnsemble(config=config, n_members=3, physics=physics)

    # Minimal training data (in-distribution)
    df = generate_pump_field_data(n_samples=100, seed=0)
    X_tr = torch.tensor(
        df[["flow_rate", "speed", "operating_hours"]].values, dtype=torch.float32
    )
    y_tr = torch.tensor(df[["head"]].values, dtype=torch.float32)
    ensemble.fit(X_tr, y_tr, max_epochs=5, batch_size=32)

    # In-distribution: normal operating point
    X_in = torch.tensor([[0.05, 1450.0, 1000.0]])
    # OOD: extreme speed far outside training range
    X_ood = torch.tensor([[0.05, 5000.0, 1000.0]])

    pred_in = ensemble.predict(X_in)
    pred_ood = ensemble.predict(X_ood)

    assert pred_ood.epistemic_std.mean() > pred_in.epistemic_std.mean(), (
        "OOD epistemic uncertainty should exceed in-distribution uncertainty"
    )
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_pinn.py::test_ensemble_predict_returns_uncertainty -v
```

Expected: `ImportError: cannot import name 'PINNEnsemble'`

- [ ] **Step 3: Implement `src/surrogates/ensemble.py`**

```python
"""Ensemble of PINNs for epistemic + aleatoric uncertainty decomposition.

Epistemic uncertainty: disagreement across ensemble members (model uncertainty)
Aleatoric uncertainty: average learned logvar (irreducible data noise)
Total uncertainty:    sqrt(epistemic² + aleatoric²)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import EarlyStopping

from src.surrogates.pinn import PINN, PINNConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.physics_models.pump import PumpPhysics


@dataclass
class EnsemblePrediction:
    mean: torch.Tensor           # [N, 1] — average prediction across members
    epistemic_std: torch.Tensor  # [N, 1] — std of member means
    aleatoric_std: torch.Tensor  # [N, 1] — sqrt of mean(exp(logvar))
    total_std: torch.Tensor      # [N, 1] — sqrt(epistemic² + aleatoric²)


class PINNEnsemble:
    """Train and predict with an ensemble of PINNs.

    Each member is trained with a different random seed, producing diverse
    predictions whose variance estimates epistemic uncertainty.
    """

    def __init__(
        self,
        config: PINNConfig,
        n_members: int = 10,
        physics: "PumpPhysics | None" = None,
    ) -> None:
        self.config = config
        self.n_members = n_members
        self.physics = physics
        self.members: list[PINN] = []

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        max_epochs: int = 300,
        batch_size: int = 64,
        early_stop_patience: int = 30,
    ) -> None:
        """Train all ensemble members. Each gets a different random seed."""
        # Store normalization stats from training data
        self._x_mean = X_train.mean(dim=0)
        self._x_std = X_train.std(dim=0) + 1e-8
        self._y_mean = y_train.mean()
        self._y_std = y_train.std() + 1e-8

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_ds = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_ds, batch_size=batch_size)

        self.members = []
        for seed in range(self.n_members):
            L.seed_everything(seed, workers=True)
            member = PINN(self.config, physics=self.physics)
            member.x_mean = self._x_mean
            member.x_std = self._x_std

            callbacks = [
                EarlyStopping(
                    monitor="val/loss_total" if val_loader else "train/loss_total",
                    patience=early_stop_patience,
                    mode="min",
                )
            ]
            trainer = L.Trainer(
                max_epochs=max_epochs,
                callbacks=callbacks,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(member, train_loader, val_loader)
            self.members.append(member)

    def predict(self, X: torch.Tensor) -> EnsemblePrediction:
        """Run all members and return decomposed uncertainty.

        Args:
            X: Input tensor [N, 3] — [flow_rate, speed, operating_hours]

        Returns:
            EnsemblePrediction with mean, epistemic_std, aleatoric_std, total_std.
        """
        if not self.members:
            raise RuntimeError("Call fit() before predict()")

        means = []
        vars_aleatoric = []

        for member in self.members:
            member.eval()
            with torch.no_grad():
                mean, logvar = member(X)
            means.append(mean)
            vars_aleatoric.append(logvar.exp())

        means_stack = torch.stack(means, dim=0)          # [M, N, 1]
        vars_stack = torch.stack(vars_aleatoric, dim=0)  # [M, N, 1]

        ensemble_mean = means_stack.mean(dim=0)           # [N, 1]
        epistemic_var = means_stack.var(dim=0)            # [N, 1]
        aleatoric_var = vars_stack.mean(dim=0)            # [N, 1]
        total_var = epistemic_var + aleatoric_var

        return EnsemblePrediction(
            mean=ensemble_mean,
            epistemic_std=epistemic_var.sqrt(),
            aleatoric_std=aleatoric_var.sqrt(),
            total_std=total_var.sqrt(),
        )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_pinn.py -v
```

Expected: all 6 tests PASS. (OOD test may be slow — ~30s for 3 members × 5 epochs.)

- [ ] **Step 5: Commit**

```bash
git add src/surrogates/ensemble.py tests/test_pinn.py
git commit -m "Add PINNEnsemble with epistemic/aleatoric uncertainty decomposition"
```

---

## Task 7: Training script

**Files:**
- Create: `scripts/train_pump_surrogate.py`

- [ ] **Step 1: Implement the training script**

```python
#!/usr/bin/env python3
"""Train PINNEnsemble on synthetic pump field data.

Usage:
    uv run python scripts/train_pump_surrogate.py
    uv run python scripts/train_pump_surrogate.py training.max_epochs=100
    uv run python scripts/train_pump_surrogate.py ensemble.n_members=3
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics_models.data_generator import generate_pump_field_data
from src.physics_models.pump import PumpParameters, PumpPhysics
from src.surrogates.ensemble import PINNEnsemble
from src.surrogates.pinn import PINNConfig

log = logging.getLogger(__name__)


def _load_or_generate_data(cfg: DictConfig) -> pd.DataFrame:
    csv_path = Path(cfg.data.csv_path)
    if csv_path.exists():
        log.info("Loading existing data from %s", csv_path)
        return pd.read_csv(csv_path)
    log.info("Generating synthetic data → %s", csv_path)
    df = generate_pump_field_data(n_samples=500, seed=cfg.seed)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def _split(df: pd.DataFrame, train_frac: float, val_frac: float):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


def _to_tensors(df: pd.DataFrame):
    X = torch.tensor(
        df[["flow_rate", "speed", "operating_hours"]].values, dtype=torch.float32
    )
    y = torch.tensor(df[["head"]].values, dtype=torch.float32)
    return X, y


@hydra.main(config_path="../configs", config_name="pump_surrogate", version_base=None)
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    df = _load_or_generate_data(cfg)
    df_train, df_val, df_test = _split(df, cfg.data.train_split, cfg.data.val_split)
    log.info("Split: %d train / %d val / %d test", len(df_train), len(df_val), len(df_test))

    X_train, y_train = _to_tensors(df_train)
    X_val, y_val = _to_tensors(df_val)
    X_test, y_test = _to_tensors(df_test)

    pump_params = PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    )
    physics = PumpPhysics(pump_params)

    pinn_config = PINNConfig(
        hidden_dims=list(cfg.model.hidden_dims),
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
        lambda_data=cfg.physics.lambda_data,
        lambda_physics=cfg.physics.lambda_physics,
        lr=cfg.training.lr,
    )

    ensemble = PINNEnsemble(
        config=pinn_config,
        n_members=cfg.ensemble.n_members,
        physics=physics,
    )

    with mlflow.start_run():
        mlflow.log_params({
            "n_members": cfg.ensemble.n_members,
            "hidden_dims": str(cfg.model.hidden_dims),
            "lambda_physics": cfg.physics.lambda_physics,
            "max_epochs": cfg.training.max_epochs,
        })

        log.info("Training ensemble of %d members...", cfg.ensemble.n_members)
        ensemble.fit(
            X_train, y_train, X_val, y_val,
            max_epochs=cfg.training.max_epochs,
            batch_size=cfg.training.batch_size,
            early_stop_patience=cfg.training.early_stop_patience,
        )

        # Evaluate on test set
        pred = ensemble.predict(X_test)
        rmse = ((pred.mean - y_test) ** 2).mean().sqrt().item()
        mean_total_std = pred.total_std.mean().item()
        # Coverage: what fraction of true values fall within ±2σ
        within_2sigma = (
            (y_test >= pred.mean - 2 * pred.total_std) &
            (y_test <= pred.mean + 2 * pred.total_std)
        ).float().mean().item()

        log.info("Test RMSE: %.3f m | Mean uncertainty: %.3f m | 2σ coverage: %.1f%%",
                 rmse, mean_total_std, within_2sigma * 100)

        mlflow.log_metrics({
            "test_rmse": rmse,
            "mean_total_std": mean_total_std,
            "coverage_2sigma": within_2sigma,
        })

        # Save ensemble
        out_dir = Path("models/pump_ensemble")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, member in enumerate(ensemble.members):
            torch.save(member.state_dict(), out_dir / f"member_{i:02d}.pt")
        log.info("Saved %d members to %s", len(ensemble.members), out_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the training script**

```bash
uv run python scripts/train_pump_surrogate.py ensemble.n_members=3 training.max_epochs=50
```

Expected output (approximate):
```
Split: 300 train / 100 val / 100 test
Training ensemble of 3 members...
Test RMSE: X.XXX m | Mean uncertainty: X.XXX m | 2σ coverage: XX.X%
Saved 3 members to models/pump_ensemble
```

Coverage should be > 80% for 2σ intervals.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_pump_surrogate.py
git commit -m "Add Hydra+MLflow training script for PINNEnsemble"
```

---

## Task 8: Full test suite and coverage check

- [ ] **Step 1: Run full test suite with coverage**

```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

Expected: all tests pass, coverage > 80% on `src/physics_models/` and `src/surrogates/`.

- [ ] **Step 2: If coverage < 80% on any module, add the missing test**

Common gap: `PumpPhysics.power()` with degraded efficiency. Add to `tests/test_pump_physics.py`:

```python
def test_power_increases_with_degraded_efficiency():
    """Lower efficiency at same operating point means higher shaft power."""
    params = PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    )
    physics_nominal = PumpPhysics(params)
    # Simulate degraded pump: manually override efficiency coefficient
    params_degraded = PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.55,  # 20% degraded
    )
    physics_degraded = PumpPhysics(params_degraded)
    p_nominal = physics_nominal.power(flow=0.04, speed=1450.0)
    p_degraded = physics_degraded.power(flow=0.04, speed=1450.0)
    assert p_degraded > p_nominal, "Degraded efficiency means higher shaft power"
```

- [ ] **Step 3: Final commit**

```bash
git add tests/
git commit -m "Ensure test coverage >80% across physics and surrogate modules"
```

---

## Self-Review

**Spec coverage:**
- [x] `pyproject.toml` with all listed dependencies: Task 1
- [x] `PumpParameters` Pydantic model with bounds checking: Task 2
- [x] `PumpPhysics` with affinity laws, head/efficiency/power: Task 2
- [x] `generate_pump_field_data()` with wear + leakage + noise + 500 rows: Task 3
- [x] Data saved to `data/pump_field_measurements.csv`: Task 3 Step 5
- [x] PINN with physics residual loss via autograd: Task 5
- [x] Heteroscedastic `uncertainty_logvar` output: Task 5
- [x] `PINNEnsemble` with `epistemic_uncertainty` / `aleatoric_uncertainty` / `total_uncertainty`: Task 6
- [x] Hydra configs `base.yaml` + `pump_surrogate.yaml` with all spec sections: Task 4
- [x] Training script with MLflow logging, model saving, metrics: Task 7
- [x] Unit tests with >80% coverage: Task 8

**Not in this plan (Week 2 plan):** Experiment notebook, conformal calibration, temperature scaling.
**Not in this plan (Week 3 plan):** FastAPI, Docker, chance-constrained optimizer.

**Placeholder scan:** None found.

**Type consistency:** `PINNConfig` defined in Task 5, consumed identically in Task 6 and Task 7. `EnsemblePrediction` dataclass defined in Task 6, returned by `PINNEnsemble.predict()` in Task 7. `PumpPhysics` defined in Task 2, passed to `PINN` and `PINNEnsemble` consistently.
