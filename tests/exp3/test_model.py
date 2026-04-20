import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from experiments.exp3_bayesian_compressor.model import BayesianResidualModel


def _make_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    phi = np.column_stack([np.ones(n), rng.uniform(0.6, 1.4, (n, 3))])
    w_true = np.array([0.02, 0.015, -0.010, 0.008])
    y = phi @ w_true + rng.normal(0, 0.005, n)
    return phi, y


def test_predict_before_fit_raises():
    model = BayesianResidualModel()
    with pytest.raises(RuntimeError, match="fit_batch"):
        model.predict(np.zeros((1, 4)))


def test_fit_batch_output_shapes():
    phi, y = _make_data()
    model = BayesianResidualModel(alpha=1.0, beta=40000.0)
    model.fit_batch(phi, y)
    mean, std = model.predict(phi)
    assert mean.shape == (200,)
    assert std.shape == (200,)


def test_fit_batch_recovers_weights():
    """With enough data and tight noise, predictions ≈ true residuals."""
    phi, y = _make_data(n=500)
    model = BayesianResidualModel(alpha=0.01, beta=40000.0)
    model.fit_batch(phi, y)
    mean, _ = model.predict(phi)
    w_true = np.array([0.02, 0.015, -0.010, 0.008])
    y_true_residual = phi @ w_true
    assert np.mean(np.abs(mean - y_true_residual)) < 0.02


def test_predict_std_positive():
    phi, y = _make_data()
    model = BayesianResidualModel(alpha=1.0, beta=40000.0)
    model.fit_batch(phi, y)
    _, std = model.predict(phi)
    assert np.all(std > 0)


def test_sequential_update_converges_to_batch():
    """Batch fit on 100 samples == sequential updates on same 100 samples."""
    phi, y = _make_data(n=100)

    batch_model = BayesianResidualModel(alpha=1.0, beta=40000.0)
    batch_model.fit_batch(phi, y)

    seq_model = BayesianResidualModel(alpha=1.0, beta=40000.0)
    seq_model.fit_batch(phi[:1], y[:1])
    for phi_i, y_i in zip(phi[1:], y[1:]):
        seq_model.update(phi_i, y_i)

    mean_batch, _ = batch_model.predict(phi)
    mean_seq, _ = seq_model.predict(phi)
    assert np.allclose(mean_batch, mean_seq, atol=1e-6)


def test_uncertainty_wider_ood():
    """OOD inputs (far from training distribution) should have larger std."""
    phi_train, y = _make_data(n=300)
    model = BayesianResidualModel(alpha=1.0, beta=40000.0)
    model.fit_batch(phi_train, y)

    phi_in = phi_train[:10]
    phi_ood = phi_train[:10].copy()
    phi_ood[:, 1:] *= 10.0

    _, std_in = model.predict(phi_in)
    _, std_ood = model.predict(phi_ood)
    assert std_ood.mean() > std_in.mean()
