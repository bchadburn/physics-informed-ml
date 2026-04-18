# experiments/exp4_surrogate_optimizer/problem.py
"""Pump network optimization problem definition.

Problem:
    Find (flow_rate, speed) to minimise pumping power P = flow_rate × head
    subject to: head >= H_MIN  (minimum delivery pressure)

The surrogate (trained PINNEnsemble) evaluates head(flow_rate, speed, operating_hours).
Operating hours is fixed at 0.0 (new pump scenario).
"""
from __future__ import annotations

import torch

# Decision variable bounds
FLOW_RATE_LB: float = 0.01   # m³/s
FLOW_RATE_UB: float = 0.10   # m³/s
SPEED_LB: float = 500.0      # RPM
SPEED_UB: float = 2500.0     # RPM

# Default constraint threshold
H_MIN_DEFAULT: float = 20.0  # metres — minimum head required


def pumping_power(flow_rate: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """Simplified pumping power: P ∝ Q × H (proportional, dimensionless)."""
    return flow_rate * head


def constraint_violation(head: torch.Tensor, h_min: float) -> torch.Tensor:
    """Amount by which head constraint is violated: max(0, h_min - head).

    Zero when feasible, positive when violated.
    """
    return torch.clamp(h_min - head, min=0.0)


def augmented_lagrangian_loss(
    flow_rate: torch.Tensor,
    head: torch.Tensor,
    h_min: float,
    penalty: float,
) -> torch.Tensor:
    """Objective + quadratic penalty for constraint violation.

    L(x) = P(x) + penalty × violation(x)²

    Args:
        flow_rate: current flow rate decision variable
        head:      predicted head from surrogate
        h_min:     minimum required head
        penalty:   augmented Lagrangian penalty coefficient
    """
    obj = pumping_power(flow_rate, head)
    viol = constraint_violation(head, h_min)
    return obj + penalty * viol ** 2


def project_to_bounds(
    flow_rate: torch.Tensor, speed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip decision variables to their feasible bounds."""
    return flow_rate.clamp(FLOW_RATE_LB, FLOW_RATE_UB), speed.clamp(SPEED_LB, SPEED_UB)
