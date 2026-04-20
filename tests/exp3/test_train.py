import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from experiments.exp3_bayesian_compressor.data import generate_compressor_data
from experiments.exp3_bayesian_compressor.model import BayesianResidualModel


def test_cold_start_then_sequential_beats_vendor():
    """After cold-start + sequential updates, Bayesian RMSE < vendor-only RMSE."""
    data = generate_compressor_data(n_samples=500, seed=0)
    mask_cold = data["t_days"] <= 30
    phi_cold = data["phi"][mask_cold]
    y_res_cold = data["eta_true"][mask_cold] - data["eta_vendor"][mask_cold]

    model = BayesianResidualModel(alpha=1.0, beta=40000.0)
    model.fit_batch(phi_cold, y_res_cold)
    for phi_i, y_i in zip(
        data["phi"][~mask_cold],
        (data["eta_true"] - data["eta_vendor"])[~mask_cold],
    ):
        model.update(phi_i, y_i)

    residual_mean, _ = model.predict(data["phi"])
    bayes_pred = data["eta_vendor"] + residual_mean

    vendor_rmse = np.sqrt(np.mean((data["eta_vendor"] - data["eta_true"]) ** 2))
    bayes_rmse = np.sqrt(np.mean((bayes_pred - data["eta_true"]) ** 2))
    assert bayes_rmse < vendor_rmse
