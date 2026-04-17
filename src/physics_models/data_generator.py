"""Synthetic pump field data generator with simulated degradation.

Simulates 500 operating measurements spanning ~1 year of pump operation.
Degradation model:
  - Impeller wear: head coefficient drops 5–15% linearly over 8760 hours
  - Seal leakage: flow measurement offset +2–8% (random per unit)
  - Measurement noise: ±3% on flow, ±5% on head (Gaussian)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.physics_models.pump import PumpParameters, PumpPhysics

# Default design-point pump (mid-range industrial centrifugal)
_DEFAULT_PARAMS = PumpParameters(
    design_flow=0.05,       # 50 L/s
    design_head=30.0,       # 30 m
    design_speed=1450.0,    # rpm
    design_efficiency=0.75,
)


def generate_pump_field_data(
    n_samples: int = 500,
    seed: int = 42,
    params: PumpParameters = _DEFAULT_PARAMS,
    max_hours: float = 8760.0,   # one year of operation
) -> pd.DataFrame:
    """Generate synthetic pump operating measurements with degradation.

    Args:
        n_samples: Number of measurement rows to generate.
        seed: Random seed for reproducibility.
        params: Pump design parameters.
        max_hours: Total operating hours span.

    Returns:
        DataFrame with columns: timestamp, flow_rate, head, speed,
        operating_hours.
    """
    rng = np.random.default_rng(seed)
    physics = PumpPhysics(params)

    # Operating hours distributed unevenly (clustered around business hours)
    operating_hours = np.sort(rng.uniform(0, max_hours, n_samples))

    # Degradation factors: progress from 0 (new) to 1 (end of life)
    degradation_progress = operating_hours / max_hours

    # Impeller wear: reduces head by 5–15% over lifetime
    wear_fraction = rng.uniform(0.05, 0.15)
    head_wear_factor = 1.0 - wear_fraction * degradation_progress

    # Seal leakage: constant flow measurement offset per pump instance
    seal_leak_fraction = rng.uniform(0.02, 0.08)

    # Speed: varies around nominal ±5% (VFD or load variation)
    speed = rng.uniform(
        params.design_speed * 0.85,
        params.design_speed * 1.05,
        n_samples,
    )

    # True flow: randomly sampled operating points across the curve
    q_true = rng.uniform(
        params.design_flow * 0.3,
        params.design_flow * 1.4,
        n_samples,
    )

    # True head from physics model + wear degradation
    h_true = np.array([
        physics.head(q, s) * hw
        for q, s, hw in zip(q_true, speed, head_wear_factor)
    ])

    # Measurements: add sensor noise
    flow_noise = rng.normal(0, 0.03, n_samples)   # ±3% relative
    head_noise = rng.normal(0, 0.05, n_samples)   # ±5% relative

    flow_measured = q_true * (1.0 + seal_leak_fraction + flow_noise)
    head_measured = h_true * (1.0 + head_noise)

    # Timestamps: 1-hour resolution from arbitrary start
    start = pd.Timestamp("2024-01-01")
    timestamps = [
        start + pd.Timedelta(hours=float(h)) for h in operating_hours
    ]

    return pd.DataFrame({
        "timestamp": timestamps,
        "flow_rate": flow_measured,
        "head": head_measured,
        "speed": speed,
        "operating_hours": operating_hours,
    })
