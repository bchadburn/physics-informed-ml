# tests/exp3/test_data.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from experiments.exp3_bayesian_compressor.data import (
    DP_RATED,
    N_RATED,
    Q_RATED,
    features,
    generate_compressor_data,
    vendor_curve,
)


def test_vendor_curve_at_rated():
    eta = vendor_curve(
        np.array([Q_RATED]), np.array([DP_RATED]), np.array([N_RATED])
    )
    assert abs(eta[0] - 0.85) < 1e-6


def test_vendor_curve_off_rated_lower():
    eta_rated = vendor_curve(
        np.array([Q_RATED]), np.array([DP_RATED]), np.array([N_RATED])
    )
    eta_off = vendor_curve(
        np.array([Q_RATED * 0.5]), np.array([DP_RATED]), np.array([N_RATED])
    )
    assert eta_off[0] < eta_rated[0]


def test_features_shape():
    Q = np.array([0.4, 0.5, 0.6])
    dP = np.array([5.0, 6.0, 7.0])
    N = np.array([3200.0, 3500.0, 3800.0])
    phi = features(Q, dP, N)
    assert phi.shape == (3, 4)
    assert np.all(phi[:, 0] == 1.0)


def test_generate_shape():
    data = generate_compressor_data(n_samples=100, seed=0)
    for key in ("Q", "dP", "N", "t_days", "eta_vendor", "eta_true", "phi"):
        assert key in data
    assert data["eta_true"].shape == (100,)
    assert data["phi"].shape == (100, 4)


def test_data_sorted_by_time():
    data = generate_compressor_data(n_samples=200, seed=0)
    assert np.all(np.diff(data["t_days"]) >= 0)


def test_efficiency_in_range():
    data = generate_compressor_data(n_samples=500, seed=0)
    assert np.all(data["eta_true"] >= 0.0)
    assert np.all(data["eta_true"] <= 1.0)


def test_drift_visible():
    """Late samples have lower mean η than early samples due to fouling."""
    data = generate_compressor_data(n_samples=2000, seed=0)
    early = data["eta_true"][data["t_days"] < 30]
    late = data["eta_true"][data["t_days"] > 335]
    assert early.mean() > late.mean()
