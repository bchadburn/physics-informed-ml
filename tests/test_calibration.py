import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from src.surrogates.calibration import ConformalCalibrator, CalibratedInterval


def _make_data(n: int, seed: int = 0):
    torch.manual_seed(seed)
    mean = torch.randn(n, 1) * 5 + 20.0
    true_std = torch.rand(n, 1) * 2 + 0.5
    y = mean + true_std * torch.randn(n, 1)
    return y, mean, true_std


def test_calibrate_returns_positive_qhat():
    y, mean, std = _make_data(200)
    cal = ConformalCalibrator()
    q_hat = cal.calibrate(y[:100], mean[:100], std[:100], coverage=0.90)
    assert q_hat > 0.0


def test_coverage_near_target():
    """Empirical coverage on held-out set should be >= target (finite-sample guarantee)."""
    y, mean, std = _make_data(1000)
    cal = ConformalCalibrator()
    cal.calibrate(y[:500], mean[:500], std[:500], coverage=0.90)
    interval = cal.predict_interval(mean[500:], std[500:])
    coverage = cal.empirical_coverage(y[500:], interval)
    assert coverage >= 0.88  # slight slack for randomness


def test_predict_interval_before_calibrate_raises():
    cal = ConformalCalibrator()
    with pytest.raises(RuntimeError, match="calibrate"):
        cal.predict_interval(torch.zeros(5, 1), torch.ones(5, 1))


def test_calibrated_interval_shape():
    y, mean, std = _make_data(100)
    cal = ConformalCalibrator()
    cal.calibrate(y, mean, std, coverage=0.90)
    interval = cal.predict_interval(mean, std)
    assert isinstance(interval, CalibratedInterval)
    assert interval.lower.shape == mean.shape
    assert interval.upper.shape == mean.shape
    assert (interval.upper >= interval.lower).all()
