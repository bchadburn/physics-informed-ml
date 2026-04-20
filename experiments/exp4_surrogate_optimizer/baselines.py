"""Random search and grid search baselines for pump optimization."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from experiments.exp4_surrogate_optimizer.problem import (
    FLOW_RATE_LB,
    FLOW_RATE_UB,
    SPEED_LB,
    SPEED_UB,
)


@dataclass
class OptimizationResult:
    """Unified result type for all optimizer variants."""
    x_opt: np.ndarray            # [flow_rate, speed]
    obj_value: float             # pumping power at x_opt
    head_at_opt: float           # predicted head at x_opt
    constraint_violation: float  # max(0, h_min - head) at x_opt
    n_surrogate_calls: int       # total calls to surrogate
    wall_time_ms: float          # wall-clock time in milliseconds


def _best_from_candidates(
    surrogate_fn: Callable,
    flow_rates: np.ndarray,
    speeds: np.ndarray,
    operating_hours: float,
    h_min: float,
) -> OptimizationResult:
    """Evaluate all candidates and return best feasible (or least-violated) point."""
    n = len(flow_rates)
    X = torch.tensor(
        np.column_stack([flow_rates, speeds, np.full(n, operating_hours)]),
        dtype=torch.float32,
    )
    with torch.no_grad():
        head = surrogate_fn(X).squeeze().cpu().numpy()

    power = flow_rates * head
    violation = np.maximum(0.0, h_min - head)

    feasible = violation == 0.0
    if feasible.any():
        idx = int(np.argmin(np.where(feasible, power, np.inf)))
    else:
        idx = int(np.argmin(violation))

    return OptimizationResult(
        x_opt=np.array([flow_rates[idx], speeds[idx]]),
        obj_value=float(power[idx]),
        head_at_opt=float(head[idx]),
        constraint_violation=float(violation[idx]),
        n_surrogate_calls=n,
        wall_time_ms=0.0,  # set by caller
    )


def random_search(
    surrogate_fn: Callable,
    h_min: float,
    operating_hours: float = 0.0,
    n_samples: int = 10000,
    seed: int = 42,
) -> OptimizationResult:
    """Random search: sample uniformly and return best feasible point."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    flow_rates = rng.uniform(FLOW_RATE_LB, FLOW_RATE_UB, n_samples)
    speeds = rng.uniform(SPEED_LB, SPEED_UB, n_samples)
    result = _best_from_candidates(surrogate_fn, flow_rates, speeds, operating_hours, h_min)
    result.wall_time_ms = (time.perf_counter() - t0) * 1000
    return result


def grid_search(
    surrogate_fn: Callable,
    h_min: float,
    operating_hours: float = 0.0,
    n_grid: int = 100,
) -> OptimizationResult:
    """Grid search over (flow_rate, speed) — serves as reference optimum.

    Uses n_grid² surrogate evaluations.
    """
    t0 = time.perf_counter()
    fr_vals = np.linspace(FLOW_RATE_LB, FLOW_RATE_UB, n_grid)
    sp_vals = np.linspace(SPEED_LB, SPEED_UB, n_grid)
    FR, SP = np.meshgrid(fr_vals, sp_vals)
    result = _best_from_candidates(
        surrogate_fn, FR.ravel(), SP.ravel(), operating_hours, h_min
    )
    result.wall_time_ms = (time.perf_counter() - t0) * 1000
    return result
