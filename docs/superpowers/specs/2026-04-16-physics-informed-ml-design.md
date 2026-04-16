# Physics-Informed ML Portfolio — Design Spec

**Goal:** A single cohesive GitHub repo demonstrating four physics-informed ML
techniques applied to industrial energy problems, benchmarked rigorously against
pure data-driven and pure physics baselines, with a capstone connecting a trained
surrogate to a gradient-based optimizer.

**Target audience:** ML Engineer interview at companies building physics-informed
AI for industrial energy (surrogate models, hybrid physics+ML, real-time
optimization). Demonstrates breadth across four distinct techniques with clear
tradeoff reasoning in each writeup.

**Timeline:** 2–3 weeks of evening/weekend work. Experiments sequenced so each
one builds conceptual foundation for the next.

---

## Repo Structure

```
physics-informed-ml/
├── README.md                        # Portfolio framing + cross-experiment results table
├── pyproject.toml                   # uv-managed: torch, numpy, scipy, matplotlib, gpytorch, h5py
├── core/
│   ├── trainer.py                   # Shared PyTorch training loop
│   ├── metrics.py                   # Shared evaluation metrics
│   └── benchmark.py                 # Baseline comparison runner
├── experiments/
│   ├── exp1_pinn_pipe_flow/
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── README.md
│   ├── exp2_fno_heat_transfer/
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── README.md
│   ├── exp3_bayesian_compressor/
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── README.md
│   └── exp4_surrogate_optimizer/
│       ├── data.py
│       ├── optimizer.py
│       ├── evaluate.py
│       └── README.md
└── notebooks/
    ├── exp1_figures.ipynb
    ├── exp2_figures.ipynb
    ├── exp3_figures.ipynb
    └── exp4_figures.ipynb
```

---

## Shared Infrastructure (`core/`)

### `core/trainer.py`

Standard PyTorch training loop used by all experiments. Interface:

```python
def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    patience: int = 20,
    checkpoint_path: Path | None = None,
) -> dict[str, list[float]]:
    """Returns dict with keys 'train_loss', 'val_loss' (lists over epochs)."""
```

Responsibilities:
- Training loop with gradient clipping (clip_grad_norm, max_norm=1.0)
- Early stopping on val loss with configurable patience
- Checkpoint save on best val loss; load from checkpoint for evaluation
- Returns loss history for plotting

### `core/metrics.py`

```python
def relative_l2_error(pred: np.ndarray, true: np.ndarray) -> float:
    """||pred - true||_2 / ||true||_2 — standard metric in PDE surrogate literature."""

def inference_timer(model: nn.Module, inputs: torch.Tensor, n_runs: int = 100) -> float:
    """Returns mean wall-clock seconds per forward pass over n_runs."""

def expected_calibration_error(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE for Bayesian model calibration (used in Exp 3)."""

def optimality_gap(surrogate_obj: float, true_obj: float) -> float:
    """(true_obj - surrogate_obj) / abs(true_obj) — used in Exp 4."""
```

### `core/benchmark.py`

```python
@dataclass
class BenchmarkResult:
    name: str
    relative_l2: float
    inference_time_ms: float
    n_train_samples: int
    notes: str = ""

def run_benchmark(
    models: dict[str, nn.Module],
    test_loader: DataLoader,
    loss_fn: Callable,
) -> list[BenchmarkResult]:
    """Run all models on test set, return results list for table rendering."""

def render_markdown_table(results: list[BenchmarkResult]) -> str:
    """Render results as a GitHub-flavored markdown table."""
```

---

## Experiment 1: PINN for Steady-State Pipe Flow

**Technique:** Physics-Informed Neural Network — PDE residual added to loss  
**Do this first:** Establishes the core PINN training pattern (loss weighting,
collocation points, boundary conditions) that underlies the whole field.

### Problem
Predict pressure drop and velocity profile across a pipe segment given inlet
velocity and fluid properties. Industrial analog: natural gas distribution
networks, cooling water systems, hydraulic circuits.

### Physics
Hagen-Poiseuille equation for laminar incompressible flow in a circular pipe:

```
ΔP = (128 μ L Q) / (π d⁴)
u(r) = (ΔP / 4μL) * (R² - r²)
```

This has a closed-form solution — ideal for validating that the PINN converges
to the correct answer, and for generating unlimited labeled data at any fidelity.

### Data (`data.py`)
Synthetic generator. Inputs: inlet velocity `u_in`, pipe radius `R`, viscosity
`μ`, pipe length `L`. Output: velocity profile `u(r)` sampled at `N` radial
points. Seeded with `np.random.seed` for reproducibility.

- Training set: N labeled (input, solution) pairs + M collocation points
  (inputs only, used for PDE residual)
- Key design: vary N from 10 to 1000 to produce data efficiency curves
- OOD test set: inlet velocities 2× outside training distribution

### Model (`model.py`)

```python
class PipeFlowPINN(nn.Module):
    """MLP with tanh activations. Input: (r, u_in, R, mu, L). Output: u(r)."""
```

Loss function:
```
L_total = L_data + λ_pde * L_pde + λ_bc * L_bc

L_data = MSE(u_pred, u_true)  # on labeled points
L_pde  = MSE(residual of H-P equation evaluated at collocation points)
L_bc   = MSE(u_pred at r=R, 0)  # no-slip boundary condition
```

λ_pde and λ_bc are hyperparameters. Ablation: fixed λ vs. learned λ via
uncertainty weighting (Kendall et al. 2018).

### Baselines
- Vanilla MLP: same architecture, trained on labeled data only (no PDE loss)
- Analytical solution: Hagen-Poiseuille closed form

### Key Metrics
- Relative L2 error vs. analytical solution
- Data efficiency curve: relative L2 vs. number of labeled training samples
  (10, 50, 100, 500, 1000)
- OOD error: relative L2 on out-of-distribution inlet velocities

### Writeup Conclusion
"The PINN matches MLP accuracy with 10× fewer labeled samples. On OOD inlet
velocities, PINN degrades gracefully (L2 error +15%) while MLP degrades sharply
(+80%). The physics constraint acts as regularization that encodes the solution
structure the MLP must learn from data alone."

Ablation to include: λ_pde sweep (0.01, 0.1, 1.0, 10.0) — shows that loss
weighting is critical and poorly understood in naive implementations.

### Background Reading
1. Raissi et al. 2019 — "Physics-informed neural networks" (the original PINN paper)
2. Wang et al. 2022 — "When and why PINNs fail to train" (failure modes, NTK analysis)
3. Clean reference: https://github.com/maziarraissi/PINNs

---

## Experiment 2: Fourier Neural Operator for Heat Transfer

**Technique:** FNO — learns the solution operator mapping function spaces  
**Do this second:** Operator learning is conceptually distinct from PINNs.
FNO requires understanding spectral convolutions; learning it after PINN
grounds it in contrast.

### Problem
Given a 2D thermal conductivity field `κ(x,y)`, predict the steady-state
temperature distribution `T(x,y)`. Industrial analog: heat exchanger design,
battery thermal management, semiconductor cooling.

### Physics
Steady-state heat equation:
```
∇·(κ(x,y) ∇T(x,y)) = f(x,y)
```
This is structurally identical to Darcy flow (`∇·(a(x)∇p(x)) = f(x)`),
which is exactly what PDEBench 2D_DarcyFlow provides.

### Data (`data.py`)
PDEBench 2D_DarcyFlow dataset. Download script included. HDF5 format,
~1GB. Loader wraps the HDF5 file into a PyTorch Dataset returning
`(conductivity_field, pressure_field)` tensors of shape `(1, 64, 64)`.

Split: 1000 train / 100 val / 100 test (standard PDEBench split).

OOD test: hold out samples where max(κ) > 95th percentile of training
distribution — tests generalization to extreme conductivity contrasts.

### Model (`model.py`)

FNO architecture (Li et al. 2021):
- 4 Fourier layers, 12 modes, channel width 32
- Each layer: spectral convolution (in frequency domain) + pointwise linear
- Final projection: channel width → 1 (scalar temperature field)

```python
class FNO2d(nn.Module):
    """
    Input:  (batch, 1, H, W) — conductivity field
    Output: (batch, 1, H, W) — temperature field
    """
```

Baseline UNet: same depth (4 encoder + 4 decoder blocks), no spectral layers,
roughly matched parameter count.

### Key Metrics
- Relative L2 error (in-distribution test set)
- Relative L2 error (OOD test set — high conductivity contrast)
- Inference time vs. FEM baseline (use FEniCS or scipy sparse solver as FEM
  reference; measure wall-clock on same hardware)
- Resolution generalization: train at 64×64, evaluate at 128×128 (FNO handles
  this; UNet does not without retraining)

### Writeup Conclusion
"FNO achieves comparable accuracy to UNet on in-distribution samples and is
3–4× faster. On OOD high-contrast conductivity fields, FNO degrades less than
UNet. At 128×128 resolution (zero-shot, trained at 64×64), FNO maintains
accuracy while UNet fails completely — this is the key production advantage:
the operator generalizes across resolutions without retraining."

### Background Reading
1. Li et al. 2021 — "Fourier Neural Operator for Parametric PDEs"
2. PDEBench paper (Takamoto et al. 2022) — for dataset context
3. Clean reference: https://github.com/neuraloperator/neuraloperator

---

## Experiment 3: Vendor-Curve-Initialized Digital Twin with Bayesian Residual

**Technique:** Physics prior (vendor curve) + Bayesian linear regression
residual with sequential posterior updates  
**Do this third:** Introduces uncertainty quantification and online learning —
concepts needed for the capstone's constraint-handling discussion.

### Problem
Predict centrifugal compressor isentropic efficiency `η` from operating
conditions `(Q, ΔP, N)` (flow rate, pressure ratio, shaft speed). Equipment
drifts from vendor specification over time due to fouling and wear. The model
must adapt to this drift and report calibrated uncertainty.

Industrial analog: gas compression stations, HVAC chillers, industrial
refrigeration — any rotating equipment with a published performance map.

### Physics Component
Vendor efficiency curve: polynomial fit to a manufacturer performance map.

```python
def vendor_curve(Q: np.ndarray, dP: np.ndarray, N: np.ndarray,
                 coeffs: np.ndarray) -> np.ndarray:
    """
    2nd-order polynomial in normalized operating variables.
    coeffs fitted once to vendor datasheet points.
    Returns predicted isentropic efficiency η_vendor.
    """
```

### Data (`data.py`)
Synthetic generator simulating 5000 operating points over a 12-month period:

```
η_true(t) = η_vendor(Q, dP, N) + β·f(Q, dP, N) + ε(t)

where:
  β·f(...)  = linear residual (3 engineered features: deviation from BEP,
               speed ratio, normalized flow)
  ε(t)      = Gaussian noise N(0, σ²)
  drift(t)  = -0.002·t/30  # linear fouling: -0.2% efficiency per month
```

Output columns: `[Q, dP, N, t, η_true, η_vendor]`

The drift term makes this a sequential learning problem.

### Model (`model.py`)

```python
class BayesianResidualModel:
    """
    Bayesian linear regression on residual: η_true - η_vendor.

    Features φ(x): [1, Q_norm, dP_norm, N_norm]  (4-dimensional)
    Prior:   w ~ N(0, α⁻¹ I)
    Likelihood: η_residual | w ~ N(φᵀw, β⁻¹)
    Posterior:  closed-form update (conjugate)

    Methods:
        fit_batch(X, y_residual): full batch posterior (initial fit)
        update(x_new, y_new):     sequential rank-1 update
        predict(X):               returns (mean, std) for each input
    """
```

Sequential update equations (closed form, no MCMC needed):

```
Posterior precision: Sₙ⁻¹ = S₀⁻¹ + β Σ φᵢφᵢᵀ
Posterior mean:      mₙ   = Sₙ(S₀⁻¹m₀ + β Σ φᵢyᵢ)
Predictive:          p(y*|x*) = N(mₙᵀφ*, φ*ᵀSₙφ* + β⁻¹)
```

### Evaluation (`evaluate.py`)
1. **Accuracy over time:** RMSE in 30-day windows — does the model track drift?
2. **Calibration:** Reliability diagram + ECE — are the uncertainty estimates
   trustworthy?
3. **Comparison:** vendor-only RMSE vs. static MLP vs. Bayesian residual at
   each time window
4. **Cold-start:** how many operating points needed before Bayesian residual
   beats vendor-only?

### Writeup Conclusion
"A static MLP trained on early data degrades as equipment fouls (+35% RMSE by
month 6). The Bayesian residual model tracks drift continuously with each new
operating point; by month 3 it has seen enough data to be well-calibrated
(ECE < 0.05). Unlike the MLP, the model knows when it's uncertain — operating
points far from the seen distribution get wider prediction intervals, which
triggers a flag rather than a silent wrong answer. This is the safe production
pattern: start from vendor knowledge, adapt with data, never be confidently wrong."

### Background Reading
1. Bishop 2006, Pattern Recognition and ML — Chapter 3.3 (Bayesian linear
   regression, closed-form posterior). Free PDF online.
2. Gal & Ghahramani 2016 — "Dropout as a Bayesian Approximation" (context for
   why calibration matters in production)
3. No reference implementation needed — Bishop Ch 3.3 is the implementation.

---

## Experiment 4: Surrogate-in-the-Loop Optimization (Capstone)

**Technique:** Differentiable surrogate (from Exp 1 or 2) inside gradient-based
optimizer via `torch.autograd`  
**Do this last:** Synthesizes everything — requires a trained surrogate,
uncertainty from Exp 3's framing, and optimization intuition from prior work.

### Problem
Optimize inlet conditions `(u_in, R_effective)` of a pipe network to minimize
pumping energy cost, subject to a minimum flow constraint at each outlet.
The surrogate (trained PINN from Exp 1) evaluates constraint feasibility.

Industrial analog: pump scheduling in water distribution or gas transmission —
find the operating point that meets demand at minimum energy.

### Connection to Prior Work
This maps directly to your OR-Tools supply chain optimizer: objectives +
constraints, just with a neural surrogate evaluating constraint values instead
of analytical functions. The key new element: the surrogate is differentiable,
so gradients flow through it — no finite-difference approximations needed.

### Optimizer (`optimizer.py`)

```python
def optimize_inlet_conditions(
    surrogate: nn.Module,
    objective_fn: Callable,      # e.g., pumping power = f(u_in, delta_P)
    constraint_fns: list[Callable],  # e.g., flow >= Q_min at each outlet
    x0: torch.Tensor,            # initial guess
    bounds: list[tuple[float, float]],
    n_steps: int = 500,
    lr: float = 1e-3,
) -> OptimizationResult:
    """
    Projected gradient descent through the surrogate.
    Returns OptimizationResult(x_opt, obj_value, constraint_violations, 
                               n_surrogate_calls, wall_time_ms).
    """
```

Constraint handling: augmented Lagrangian — penalty term added to objective
for constraint violations, penalty weight increased over iterations.

### Baselines
1. **Random search:** 10,000 random samples of inlet conditions, pick best
   feasible point — establishes floor
2. **Grid search:** systematic grid over inlet space — establishes reference
   optimum (ground truth for optimality gap)
3. **Simulator-in-loop:** same optimizer but calls analytical H-P equation
   directly instead of surrogate

### Key Metrics
- Optimality gap: `(grid_obj - surrogate_obj) / grid_obj`
- Constraint violation rate: fraction of returned solutions that violate
  physical constraints when checked against true simulator
- Surrogate call count vs. simulator call count to reach same objective value
- Wall-clock time: surrogate-in-loop vs. simulator-in-loop

### Writeup Conclusion
"Surrogate-in-loop finds solutions within 3% of the true optimum using 500×
fewer simulator evaluations. The key failure mode: surrogate prediction error
near constraint boundaries causes ~8% of returned solutions to violate physical
constraints when checked. Production mitigation: add a safety margin to
constraint bounds (conservative feasibility), or use the Bayesian residual
uncertainty from Exp 3 to tighten bounds adaptively where the surrogate is
uncertain. This connects the four experiments: PINN gives the fast surrogate,
Bayesian residual gives the uncertainty estimate, and the optimizer uses both."

### Background Reading
1. Lütjens et al. 2021 — "Physics-informed surrogate models for optimization"
2. Nocedal & Wright, "Numerical Optimization" Ch 17 (augmented Lagrangian) —
   for constraint handling theory
3. No reference implementation — the optimizer is straightforward once you
   have the surrogate.

---

## Cross-Cutting Decisions

### PyTorch conventions
- All models inherit `nn.Module`
- All data pipelines return `torch.utils.data.Dataset`
- Random seeds set via `torch.manual_seed(42)` + `np.random.seed(42)` in every
  `train.py`
- Configs as Python dataclasses (no YAML/Hydra — YAGNI for a portfolio project)

### Dependency management
`uv` with `pyproject.toml`. Core dependencies:
- `torch >= 2.2`
- `numpy`, `scipy`, `matplotlib`
- `h5py` (PDEBench HDF5 loader)
- `gpytorch` (available if BLR insufficient — not used in current design)
- `tqdm` (training progress)

### What this portfolio demonstrates
- PINN: physics as regularization, data efficiency tradeoff
- FNO: operator learning, resolution generalization, OOD behavior
- Bayesian hybrid: uncertainty quantification, online adaptation, production safety
- Surrogate optimizer: synthesis — trained model deployed inside decision loop

### Known rabbit holes to avoid
- **PINN:** Don't chase perfect convergence. Loss balancing (λ_pde tuning) can
  consume days. Time-box the ablation to 4 values and move on.
- **FNO:** Don't implement spectral convolutions from scratch. Use the
  `neuraloperator` library, understand it, then document why you used it.
- **Exp 3:** Don't extend to full GP. Bayesian linear regression delivers the
  same interview story (uncertainty, calibration, online updates) in 2 hours
  of implementation vs. 2 days for GP.
- **Exp 4:** Don't over-engineer the optimizer. Projected gradient descent is
  sufficient. The story is about surrogate + optimization, not optimizer design.
