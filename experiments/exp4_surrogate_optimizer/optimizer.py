"""Projected gradient descent with augmented Lagrangian constraint handling.

Decision variables (flow_rate, speed) are normalised to [0, 1] for numerical
stability. After each Adam step they are clamped to [0, 1], which is equivalent
to projection onto the feasible box in the original space.

The surrogate is called without torch.no_grad() so gradients flow back through
the neural network to the input tensors.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch

from experiments.exp4_surrogate_optimizer.baselines import OptimizationResult
from experiments.exp4_surrogate_optimizer.problem import (
    FLOW_RATE_LB,
    FLOW_RATE_UB,
    SPEED_LB,
    SPEED_UB,
    augmented_lagrangian_loss,
)


def optimize_multistart(
    surrogate_fn: Callable,
    h_min: float,
    operating_hours: float = 0.0,
    n_starts: int = 10,
    seed: int = 0,
    **kwargs,
) -> OptimizationResult:
    """Run optimize() from multiple random starts and return the best result.

    Mitigates non-convexity: a single gradient run can converge to a local
    minimum far from the global optimum. Keeping the best feasible result
    across N diverse starts significantly improves solution quality.

    Args:
        n_starts: number of random starting points
        seed:     RNG seed for reproducible starts
        **kwargs: forwarded to optimize() (n_steps, lr, penalty_weight)
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_starts):
        x0 = np.array([
            rng.uniform(FLOW_RATE_LB, FLOW_RATE_UB),
            rng.uniform(SPEED_LB, SPEED_UB),
        ])
        results.append(optimize(surrogate_fn, h_min, operating_hours, x0=x0, **kwargs))

    feasible = [r for r in results if r.constraint_violation == 0.0]
    best = min(feasible if feasible else results, key=lambda r: r.obj_value)

    # Report aggregated cost across all starts
    best.n_surrogate_calls = sum(r.n_surrogate_calls for r in results)
    best.wall_time_ms = sum(r.wall_time_ms for r in results)
    return best


def optimize(
    surrogate_fn: Callable,
    h_min: float,
    operating_hours: float = 0.0,
    x0: np.ndarray | None = None,
    n_steps: int = 500,
    lr: float = 1e-2,
    penalty_weight: float = 100.0,
) -> OptimizationResult:
    """Projected gradient descent through the surrogate.

    Args:
        surrogate_fn:    callable (1, 3) → (1, 1) — must support autograd
        h_min:           minimum required head (constraint threshold)
        operating_hours: fixed operating hours input to surrogate
        x0:              initial [flow_rate, speed] (random if None)
        n_steps:         number of gradient descent iterations
        lr:              Adam learning rate on normalised [0, 1] parameters
        penalty_weight:  augmented Lagrangian penalty coefficient

    Returns:
        OptimizationResult with optimised x, objective value, and diagnostics.
    """
    t0 = time.perf_counter()

    if x0 is None:
        rng = np.random.default_rng(0)
        x0 = np.array([
            rng.uniform(FLOW_RATE_LB, FLOW_RATE_UB),
            rng.uniform(SPEED_LB, SPEED_UB),
        ])

    # Normalise to [0, 1] for Adam stability
    fr_norm = torch.tensor(
        [(x0[0] - FLOW_RATE_LB) / (FLOW_RATE_UB - FLOW_RATE_LB)],
        dtype=torch.float32, requires_grad=True,
    )
    sp_norm = torch.tensor(
        [(x0[1] - SPEED_LB) / (SPEED_UB - SPEED_LB)],
        dtype=torch.float32, requires_grad=True,
    )

    adam = torch.optim.Adam([fr_norm, sp_norm], lr=lr)
    n_calls = 0

    for _ in range(n_steps):
        adam.zero_grad()

        # De-normalise to physical units
        fr = fr_norm * (FLOW_RATE_UB - FLOW_RATE_LB) + FLOW_RATE_LB
        sp = sp_norm * (SPEED_UB - SPEED_LB) + SPEED_LB

        X = torch.stack([fr, sp, torch.tensor([operating_hours])], dim=1)
        head = surrogate_fn(X)  # (1, 1) — gradients flow through here
        n_calls += 1

        loss = augmented_lagrangian_loss(fr, head.squeeze(), h_min, penalty_weight)
        loss.backward()
        adam.step()

        # Project normalised params back to [0, 1]
        with torch.no_grad():
            fr_norm.clamp_(0.0, 1.0)
            sp_norm.clamp_(0.0, 1.0)

    # Final evaluation at converged point
    with torch.no_grad():
        fr_final = float(fr_norm * (FLOW_RATE_UB - FLOW_RATE_LB) + FLOW_RATE_LB)
        sp_final = float(sp_norm * (SPEED_UB - SPEED_LB) + SPEED_LB)
        X_final = torch.tensor([[fr_final, sp_final, operating_hours]], dtype=torch.float32)
        head_final = float(surrogate_fn(X_final).squeeze())
        n_calls += 1

    return OptimizationResult(
        x_opt=np.array([fr_final, sp_final]),
        obj_value=fr_final * head_final,
        head_at_opt=head_final,
        constraint_violation=max(0.0, h_min - head_final),
        n_surrogate_calls=n_calls,
        wall_time_ms=(time.perf_counter() - t0) * 1000,
    )
