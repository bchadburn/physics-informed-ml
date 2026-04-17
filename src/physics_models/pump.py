"""Centrifugal pump physics model using affinity laws and parabolic curves.

Reference: Grundfos Pump Handbook, Chapter 1 (pump curve fundamentals).
Affinity laws: Q ∝ N, H ∝ N², P ∝ N³
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, field_validator


class PumpParameters(BaseModel):
    """Vendor-supplied design point and curve shape for a centrifugal pump."""

    design_flow: float        # m³/s at best efficiency point (BEP)
    design_head: float        # m at BEP
    design_speed: float       # rpm at BEP
    design_efficiency: float  # dimensionless [0, 1] at BEP

    @field_validator("design_flow", "design_head", "design_speed")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Must be positive, got {v}")
        return v

    @field_validator("design_efficiency")
    @classmethod
    def efficiency_in_range(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError(f"Efficiency must be in (0, 1), got {v}")
        return v


class PumpPhysics:
    """Physics model for a centrifugal pump.

    Head-flow curve:    H(Q) = a - b*Q²   (parabolic, fitted to design point)
    Efficiency curve:   η(Q) = c*Q - d*Q² (parabolic, peak at BEP)
    Affinity laws scale head and flow with speed ratio.
    """

    def __init__(self, params: PumpParameters) -> None:
        self.params = params
        # Fit parabolic coefficients so the curve passes through:
        #   H(0) = shutoff head = 1.33 * design_head (typical)
        #   H(design_flow) = design_head
        Q0 = params.design_flow
        H0 = params.design_head
        self._a = 1.33 * H0                    # shutoff head
        self._b = (self._a - H0) / (Q0 ** 2)  # curvature

        # Efficiency curve: η(Q) = c*Q - d*Q²
        # Constraints: η(Q0) = design_efficiency, dη/dQ|Q0 = 0 (peak at BEP)
        # From dη/dQ = c - 2d*Q0 = 0 → c = 2*d*Q0
        # From η(Q0) = c*Q0 - d*Q0² = d*Q0² → d = design_efficiency / Q0²
        self._d = params.design_efficiency / (Q0 ** 2)
        self._c = 2 * self._d * Q0

    def _scale_to_speed(self, flow: float, speed: float) -> tuple[float, float]:
        """Apply affinity laws: scale flow and head reference to given speed."""
        ratio = speed / self.params.design_speed
        q_ref = flow / ratio          # equivalent flow at design speed
        return q_ref, ratio

    def head(self, flow: float, speed: float) -> float:
        """Predict head [m] at given flow [m³/s] and speed [rpm]."""
        if flow < 0:
            raise ValueError(f"flow must be >= 0, got {flow}")
        q_ref, ratio = self._scale_to_speed(flow, speed)
        h_ref = self._a - self._b * q_ref ** 2
        return h_ref * ratio ** 2

    def efficiency(self, flow: float, speed: float) -> float:
        """Predict isentropic efficiency [0, 1] at given flow and speed."""
        if flow < 0:
            raise ValueError(f"flow must be >= 0, got {flow}")
        q_ref, _ = self._scale_to_speed(flow, speed)
        eta = self._c * q_ref - self._d * q_ref ** 2
        return float(np.clip(eta, 0.0, 1.0))

    def power(self, flow: float, speed: float, rho: float = 1000.0) -> float:
        """Predict shaft power [W] at given flow and speed.

        P = rho * g * Q * H / eta
        rho: fluid density [kg/m³], default water at 20°C
        """
        if flow < 0:
            raise ValueError(f"flow must be >= 0, got {flow}")
        g = 9.81
        h = self.head(flow, speed)
        eta = self.efficiency(flow, speed)
        eta = max(eta, 1e-6)  # avoid division by zero at zero flow
        return rho * g * flow * h / eta
