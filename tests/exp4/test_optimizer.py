import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from experiments.exp4_surrogate_optimizer.baselines import OptimizationResult
from experiments.exp4_surrogate_optimizer.optimizer import optimize, optimize_multistart
from experiments.exp4_surrogate_optimizer.problem import (
    FLOW_RATE_LB,
    FLOW_RATE_UB,
    H_MIN_DEFAULT,
    SPEED_LB,
    SPEED_UB,
)


def _mock_surrogate(X: torch.Tensor) -> torch.Tensor:
    """Differentiable analytical surrogate: head = 40×speed_norm − 10×flow_norm."""
    flow_norm = X[:, 0:1] / 0.05
    speed_norm = X[:, 1:2] / 1500.0
    return 40.0 * speed_norm - 10.0 * flow_norm


def test_optimize_returns_result():
    result = optimize(_mock_surrogate, h_min=H_MIN_DEFAULT, n_steps=50)
    assert isinstance(result, OptimizationResult)
    assert result.wall_time_ms > 0
    assert result.n_surrogate_calls > 0


def test_optimize_respects_bounds():
    result = optimize(_mock_surrogate, h_min=H_MIN_DEFAULT, n_steps=200)
    assert FLOW_RATE_LB <= result.x_opt[0] <= FLOW_RATE_UB
    assert SPEED_LB <= result.x_opt[1] <= SPEED_UB


def test_optimize_finds_near_feasible():
    """After 300 steps, constraint violation should be < 2 m."""
    result = optimize(_mock_surrogate, h_min=H_MIN_DEFAULT, n_steps=300, lr=5e-3)
    assert result.constraint_violation < 2.0


def test_optimize_fewer_calls_than_grid():
    """Gradient optimizer uses far fewer calls than a 100×100 grid (10000)."""
    result = optimize(_mock_surrogate, h_min=H_MIN_DEFAULT, n_steps=500)
    assert result.n_surrogate_calls < 10000


def test_optimize_custom_x0():
    x0 = np.array([0.05, 1500.0])
    result = optimize(_mock_surrogate, h_min=H_MIN_DEFAULT, x0=x0, n_steps=50)
    assert isinstance(result, OptimizationResult)


def test_multistart_returns_result():
    result = optimize_multistart(_mock_surrogate, h_min=H_MIN_DEFAULT, n_starts=3, n_steps=50)
    assert isinstance(result, OptimizationResult)
    assert FLOW_RATE_LB - 1e-5 <= result.x_opt[0] <= FLOW_RATE_UB + 1e-5
    assert SPEED_LB - 1e-1 <= result.x_opt[1] <= SPEED_UB + 1e-1


def test_multistart_total_calls_equals_sum_of_starts():
    n_starts = 3
    n_steps = 50
    result = optimize_multistart(
        _mock_surrogate, h_min=H_MIN_DEFAULT, n_starts=n_starts, n_steps=n_steps
    )
    # Each run uses n_steps + 1 final eval calls; total = n_starts * (n_steps + 1)
    assert result.n_surrogate_calls == n_starts * (n_steps + 1)


def test_multistart_returns_lower_or_equal_obj_than_best_single_run():
    """Multi-start picks the best of N runs, so its obj ≤ any individual run."""
    runs = [optimize(_mock_surrogate, h_min=H_MIN_DEFAULT, n_steps=200) for _ in range(5)]
    best_single = min(r.obj_value for r in runs)
    multi = optimize_multistart(_mock_surrogate, h_min=H_MIN_DEFAULT, n_starts=5, n_steps=200)
    assert multi.obj_value <= best_single + 1e-4
