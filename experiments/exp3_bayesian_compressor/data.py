"""Synthetic compressor operating data with linear fouling drift."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Rated operating conditions
Q_RATED: float = 0.5      # m³/s
DP_RATED: float = 6.0     # bar
N_RATED: float = 3500.0   # RPM

NOISE_STD: float = 0.005  # efficiency measurement noise (std)

DAYS_PER_MONTH: float = 30.0
_ETA_PEAK: float = 0.85
_ETA_COEFF_Q: float = -0.40
_ETA_COEFF_DP: float = -0.15
_ETA_COEFF_N: float = -0.25

# True residual weights (unknown to the model — used only in data generation)
_BETA_TRUE = np.array([0.02, 0.015, -0.010, 0.008])


def vendor_curve(Q: np.ndarray, dP: np.ndarray, N: np.ndarray) -> np.ndarray:
    """2nd-order polynomial vendor efficiency map. Peak η=0.85 at rated conditions."""
    Q_n = Q / Q_RATED
    dP_n = dP / DP_RATED
    N_n = N / N_RATED
    return (
        _ETA_PEAK
        + _ETA_COEFF_Q * (Q_n - 1.0) ** 2
        + _ETA_COEFF_N * (N_n - 1.0) ** 2
        + _ETA_COEFF_DP * (dP_n - 1.0) ** 2
    )


def features(Q: np.ndarray, dP: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Build 4-dim Bayesian feature matrix: [1, Q_norm, dP_norm, N_norm]."""
    Q_n = Q / Q_RATED
    dP_n = dP / DP_RATED
    N_n = N / N_RATED
    return np.column_stack([np.ones(len(Q)), Q_n, dP_n, N_n])


def generate_compressor_data(n_samples: int = 5000, seed: int = 42) -> dict:
    """Generate synthetic operating data over ~12 months with linear fouling drift.

    η_true(t) = η_vendor(Q, dP, N) + φ(x)ᵀ β_true + drift(t) + ε
    drift(t)  = -0.002 * t_days / 30   (-0.2% per month)
    ε         ~ N(0, NOISE_STD²)
    """
    rng = np.random.default_rng(seed)

    t_days = np.sort(rng.uniform(0, 365, n_samples))
    Q = rng.uniform(0.3 * Q_RATED, 1.7 * Q_RATED, n_samples)
    dP = rng.uniform(0.5 * DP_RATED, 1.5 * DP_RATED, n_samples)
    N = rng.uniform(0.7 * N_RATED, 1.3 * N_RATED, n_samples)

    eta_vendor = vendor_curve(Q, dP, N)
    drift = -0.002 * t_days / DAYS_PER_MONTH
    phi = features(Q, dP, N)
    residual = phi @ _BETA_TRUE
    noise = rng.normal(0.0, NOISE_STD, n_samples)
    eta_true = np.clip(eta_vendor + residual + drift + noise, 0.0, 1.0)

    log.info("Generated %d operating points over %.0f days", n_samples, t_days.max())

    return {
        "Q": Q, "dP": dP, "N": N,
        "t_days": t_days,
        "eta_vendor": eta_vendor,
        "eta_true": eta_true,
        "phi": phi,
    }
