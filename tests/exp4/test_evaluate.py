import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from core.metrics import optimality_gap
from experiments.exp4_surrogate_optimizer.baselines import OptimizationResult


def test_optimality_gap_positive_when_worse():
    gap = optimality_gap(1.1, 1.0)
    assert gap == pytest.approx(0.1)


def test_optimality_gap_negative_when_better():
    gap = optimality_gap(0.9, 1.0)
    assert gap == pytest.approx(-0.1)


def test_optimization_result_fields():
    r = OptimizationResult(
        x_opt=np.array([0.05, 1500.0]),
        obj_value=1.5,
        head_at_opt=30.0,
        constraint_violation=0.0,
        n_surrogate_calls=501,
        wall_time_ms=5.0,
    )
    assert r.x_opt.shape == (2,)
    assert r.constraint_violation == 0.0
    assert r.n_surrogate_calls == 501
