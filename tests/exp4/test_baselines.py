import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from experiments.exp4_surrogate_optimizer.baselines import (
    OptimizationResult,
    grid_search,
    random_search,
)
from experiments.exp4_surrogate_optimizer.problem import (
    FLOW_RATE_LB,
    FLOW_RATE_UB,
    H_MIN_DEFAULT,
    SPEED_LB,
    SPEED_UB,
)


def _mock_surrogate(X: torch.Tensor) -> torch.Tensor:
    """Analytical surrogate: head = 40×speed_norm − 10×flow_norm."""
    flow_norm = X[:, 0] / 0.05
    speed_norm = X[:, 1] / 1500.0
    head = 40.0 * speed_norm - 10.0 * flow_norm
    return head.unsqueeze(1)


def test_random_search_returns_result():
    result = random_search(_mock_surrogate, h_min=H_MIN_DEFAULT, n_samples=200, seed=0)
    assert isinstance(result, OptimizationResult)
    assert result.n_surrogate_calls == 200
    assert result.wall_time_ms > 0


def test_random_search_respects_bounds():
    result = random_search(_mock_surrogate, h_min=H_MIN_DEFAULT, n_samples=500, seed=0)
    assert FLOW_RATE_LB <= result.x_opt[0] <= FLOW_RATE_UB
    assert SPEED_LB <= result.x_opt[1] <= SPEED_UB


def test_grid_search_call_count():
    result = grid_search(_mock_surrogate, h_min=H_MIN_DEFAULT, n_grid=10)
    assert result.n_surrogate_calls == 100  # 10×10


def test_grid_search_finds_feasible_point():
    result = grid_search(_mock_surrogate, h_min=H_MIN_DEFAULT, n_grid=20)
    assert result.constraint_violation == 0.0


def test_grid_search_result_fields():
    result = grid_search(_mock_surrogate, h_min=H_MIN_DEFAULT, n_grid=10)
    assert isinstance(result.x_opt, np.ndarray)
    assert result.x_opt.shape == (2,)
    assert result.wall_time_ms > 0
    assert result.head_at_opt > 0
