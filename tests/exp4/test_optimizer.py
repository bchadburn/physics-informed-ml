import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
import torch
from experiments.exp4_surrogate_optimizer.optimizer import optimize
from experiments.exp4_surrogate_optimizer.baselines import OptimizationResult
from experiments.exp4_surrogate_optimizer.problem import (
    H_MIN_DEFAULT, FLOW_RATE_LB, FLOW_RATE_UB, SPEED_LB, SPEED_UB,
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
