# tests/test_pump_physics.py
import numpy as np
import pytest
from src.physics_models.pump import PumpParameters, PumpPhysics


def test_head_at_design_point():
    """At design flow and speed, predicted head should match design head."""
    params = PumpParameters(
        design_flow=0.05,       # m³/s
        design_head=30.0,       # m
        design_speed=1450.0,    # rpm
        design_efficiency=0.75,
    )
    physics = PumpPhysics(params)
    h = physics.head(flow=0.05, speed=1450.0)
    assert abs(h - 30.0) < 1.0, f"Expected ~30m, got {h:.2f}m"


def test_affinity_law_head_scales_with_speed_squared():
    """Doubling speed should quadruple head (affinity law)."""
    params = PumpParameters(
        design_flow=0.05,
        design_head=30.0,
        design_speed=1450.0,
        design_efficiency=0.75,
    )
    physics = PumpPhysics(params)
    h1 = physics.head(flow=0.03, speed=1450.0)
    h2 = physics.head(flow=0.06, speed=2900.0)  # 2x speed, 2x flow (affinity)
    ratio = h2 / h1
    assert abs(ratio - 4.0) < 0.5, f"Expected head ratio ~4, got {ratio:.2f}"


def test_efficiency_peak_near_design_flow():
    """Efficiency should be highest near design flow."""
    params = PumpParameters(
        design_flow=0.05,
        design_head=30.0,
        design_speed=1450.0,
        design_efficiency=0.75,
    )
    physics = PumpPhysics(params)
    eta_design = physics.efficiency(flow=0.05, speed=1450.0)
    eta_low = physics.efficiency(flow=0.01, speed=1450.0)
    eta_high = physics.efficiency(flow=0.09, speed=1450.0)
    assert eta_design > eta_low
    assert eta_design > eta_high


def test_power_positive():
    """Hydraulic power must always be positive for positive flow."""
    params = PumpParameters(
        design_flow=0.05,
        design_head=30.0,
        design_speed=1450.0,
        design_efficiency=0.75,
    )
    physics = PumpPhysics(params)
    flows = np.linspace(0.01, 0.09, 20)
    for q in flows:
        p = physics.power(flow=q, speed=1450.0)
        assert p > 0, f"Power must be positive, got {p:.2f} at flow={q:.3f}"


def test_invalid_flow_raises():
    """Negative flow should raise ValueError."""
    params = PumpParameters(
        design_flow=0.05,
        design_head=30.0,
        design_speed=1450.0,
        design_efficiency=0.75,
    )
    physics = PumpPhysics(params)
    with pytest.raises(ValueError, match="flow"):
        physics.head(flow=-0.01, speed=1450.0)
