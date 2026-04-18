# Experiment 3: Bayesian Residual Compressor Digital Twin

**Technique:** Physics prior (vendor curve) + Bayesian linear regression residual with sequential posterior updates

## Problem

Predict centrifugal compressor isentropic efficiency `η` from operating conditions `(Q, dP, N)`.
Equipment fouls at -0.2% efficiency/month. The model must:
1. Start from a vendor-published efficiency curve (no ML cold start)
2. Learn the true residual (vendor curve error) from operating data
3. Adapt to fouling drift continuously with each new measurement
4. Report calibrated uncertainty — widen intervals where data is sparse

## Physics

Vendor efficiency map: 2nd-order polynomial in normalized `(Q, dP, N)`, peak η=0.85 at rated conditions.
True efficiency = vendor curve + linear residual + fouling drift + noise.

## Run

```bash
# Train
uv run python experiments/exp3_bayesian_compressor/train.py

# Evaluate (drift tracking, calibration, cold-start analysis)
uv run python experiments/exp3_bayesian_compressor/evaluate.py

# Override defaults
uv run python experiments/exp3_bayesian_compressor/train.py data.cold_start_days=60
```

## Model

Bayesian linear regression (Bishop 2006, Ch 3.3) on the residual `η_true − η_vendor`:

- **Features** φ(x): `[1, Q_norm, dP_norm, N_norm]` (4-dimensional)
- **Prior**: w ~ N(0, α⁻¹ I)
- **Likelihood**: η_residual | w ~ N(φᵀw, β⁻¹)
- **Posterior**: closed-form conjugate update — no MCMC, no approximation

**Sequential rank-1 update** — O(d²) per observation, suitable for streaming deployment:

```
S_N_new⁻¹ = S_N⁻¹ + β φ_new φ_newᵀ
m_N_new   = S_N_new (S_N⁻¹ m_N + β φ_new y_new)
```

## Key Results

See `results.md` after running evaluate.py.
