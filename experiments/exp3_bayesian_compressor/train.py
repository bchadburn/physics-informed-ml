#!/usr/bin/env python3
"""Fit Bayesian residual model on compressor data with sequential updates.

Usage:
    uv run python experiments/exp3_bayesian_compressor/train.py
    uv run python experiments/exp3_bayesian_compressor/train.py data.cold_start_days=60
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.exp3_bayesian_compressor.data import generate_compressor_data
from experiments.exp3_bayesian_compressor.model import BayesianResidualModel

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="bayesian_compressor", version_base=None)
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    data = generate_compressor_data(n_samples=cfg.data.n_samples, seed=cfg.seed)

    mask_cold = data["t_days"] <= cfg.data.cold_start_days
    phi_cold = data["phi"][mask_cold]
    y_res_cold = data["eta_true"][mask_cold] - data["eta_vendor"][mask_cold]

    model = BayesianResidualModel(alpha=cfg.model.alpha, beta=cfg.model.beta)
    model.fit_batch(phi_cold, y_res_cold)
    log.info("Cold-start fit: %d samples (first %d days)", mask_cold.sum(), cfg.data.cold_start_days)

    phi_online = data["phi"][~mask_cold]
    y_res_online = (data["eta_true"] - data["eta_vendor"])[~mask_cold]
    for phi_i, y_i in zip(phi_online, y_res_online):
        model.update(phi_i, y_i)
    log.info("Sequential updates: %d observations", (~mask_cold).sum())

    residual_mean, residual_std = model.predict(data["phi"])
    bayes_pred = data["eta_vendor"] + residual_mean

    vendor_rmse = float(np.sqrt(np.mean((data["eta_vendor"] - data["eta_true"]) ** 2)))
    bayes_rmse = float(np.sqrt(np.mean((bayes_pred - data["eta_true"]) ** 2)))
    mean_std = float(residual_std.mean())

    log.info(
        "Vendor-only RMSE: %.4f | Bayesian residual RMSE: %.4f | Mean std: %.4f",
        vendor_rmse, bayes_rmse, mean_std,
    )

    with mlflow.start_run():
        mlflow.log_params({
            "n_samples": cfg.data.n_samples,
            "cold_start_days": cfg.data.cold_start_days,
            "alpha": cfg.model.alpha,
            "beta": cfg.model.beta,
        })
        mlflow.log_metrics({
            "vendor_rmse": vendor_rmse,
            "bayes_rmse": bayes_rmse,
            "mean_predictive_std": mean_std,
        })


if __name__ == "__main__":
    main()
