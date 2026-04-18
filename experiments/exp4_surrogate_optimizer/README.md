# Experiment 4: Surrogate-in-the-Loop Optimization (Capstone)

**Technique:** Differentiable surrogate (PINNEnsemble from Week 1) inside gradient-based optimizer via `torch.autograd`

## Problem

Find pump operating conditions `(flow_rate, speed)` that minimise pumping power:

```
minimise   P = flow_rate × head(flow_rate, speed)
subject to head ≥ H_min  (minimum delivery pressure)
```

## Run

```bash
uv run python experiments/exp4_surrogate_optimizer/evaluate.py
uv run python experiments/exp4_surrogate_optimizer/evaluate.py optimization.h_min=25.0
```

## How it works

Call surrogate members **without** `torch.no_grad()`:

```python
mean, _ = member(X)   # gradients flow through the network to X
loss = pumping_power(flow_rate, mean) + penalty * constraint_violation(mean, h_min) ** 2
loss.backward()       # dL/d(flow_rate), dL/d(speed) via autograd
```

Decision variables normalised to [0, 1] for Adam stability; clamped after each step for bounds projection. Constraint handled via augmented Lagrangian: `L = P + ρ × max(0, H_min − H)²`.

## Key Results

See `results.md` after running evaluate.py.
