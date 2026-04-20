import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from experiments.exp3_bayesian_compressor.evaluate import rmse_in_windows


def test_rmse_in_windows_shape():
    t = np.linspace(0, 365, 1000)
    y_true = np.zeros(1000)
    y_pred = np.ones(1000) * 0.1
    centers, rmses = rmse_in_windows(t, y_true, y_pred, window_days=30)
    assert len(centers) == len(rmses)
    assert len(centers) >= 10


def test_rmse_in_windows_values():
    t = np.linspace(0, 365, 1000)
    y_true = np.zeros(1000)
    y_pred = np.ones(1000) * 0.1
    _, rmses = rmse_in_windows(t, y_true, y_pred, window_days=30)
    assert np.allclose(rmses, 0.1, atol=1e-9)


def test_rmse_in_windows_drift():
    t = np.linspace(0, 365, 1000)
    y_true = np.zeros(1000)
    y_pred = t / 365.0 * 0.1
    centers, rmses = rmse_in_windows(t, y_true, y_pred, window_days=30)
    assert rmses[-1] > rmses[0]
