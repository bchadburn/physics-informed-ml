# tests/exp4/test_problem.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch

from core.metrics import optimality_gap
from experiments.exp4_surrogate_optimizer.problem import (
    FLOW_RATE_LB,
    SPEED_UB,
    augmented_lagrangian_loss,
    constraint_violation,
    project_to_bounds,
    pumping_power,
)


def test_pumping_power_value():
    fr = torch.tensor([0.05])
    h = torch.tensor([30.0])
    assert pumping_power(fr, h).item() == pytest.approx(1.5)


def test_constraint_violation_feasible():
    assert constraint_violation(torch.tensor([25.0]), h_min=20.0).item() == 0.0


def test_constraint_violation_infeasible():
    assert constraint_violation(torch.tensor([15.0]), h_min=20.0).item() == pytest.approx(5.0)


def test_augmented_lagrangian_feasible_equals_power():
    fr = torch.tensor([0.05])
    head = torch.tensor([30.0])
    loss = augmented_lagrangian_loss(fr, head, h_min=20.0, penalty=100.0)
    assert loss.item() == pytest.approx(pumping_power(fr, head).item())


def test_augmented_lagrangian_infeasible_exceeds_power():
    fr = torch.tensor([0.05])
    head = torch.tensor([10.0])  # below h_min=20
    loss_al = augmented_lagrangian_loss(fr, head, h_min=20.0, penalty=100.0)
    assert loss_al.item() > pumping_power(fr, head).item()


def test_project_to_bounds_clips():
    fr, sp = project_to_bounds(torch.tensor([-1.0]), torch.tensor([9999.0]))
    assert fr.item() == pytest.approx(FLOW_RATE_LB)
    assert sp.item() == pytest.approx(SPEED_UB)


def test_optimality_gap_zero_when_equal():
    assert optimality_gap(1.0, 1.0) == pytest.approx(0.0)


def test_optimality_gap_ten_percent():
    assert optimality_gap(1.1, 1.0) == pytest.approx(0.1)
