#!/usr/bin/env python3
"""Train PINNEnsemble on synthetic pump field data.

Usage:
    uv run python scripts/train_pump_surrogate.py
    uv run python scripts/train_pump_surrogate.py training.max_epochs=100
    uv run python scripts/train_pump_surrogate.py ensemble.n_members=3
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics_models.data_generator import generate_pump_field_data
from src.physics_models.pump import PumpParameters, PumpPhysics
from src.surrogates.ensemble import PINNEnsemble
from src.surrogates.pinn import PINNConfig

log = logging.getLogger(__name__)


def _load_or_generate_data(cfg: DictConfig) -> pd.DataFrame:
    csv_path = Path(cfg.data.csv_path)
    if csv_path.exists():
        log.info("Loading existing data from %s", csv_path)
        return pd.read_csv(csv_path)
    log.info("Generating synthetic data → %s", csv_path)
    df = generate_pump_field_data(n_samples=500, seed=cfg.seed)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def _split(df: pd.DataFrame, train_frac: float, val_frac: float):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


def _to_tensors(df: pd.DataFrame):
    X = torch.tensor(
        df[["flow_rate", "speed", "operating_hours"]].values, dtype=torch.float32
    )
    y = torch.tensor(df[["head"]].values, dtype=torch.float32)
    return X, y


@hydra.main(config_path="../configs", config_name="pump_surrogate", version_base=None)
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    df = _load_or_generate_data(cfg)
    df_train, df_val, df_test = _split(df, cfg.data.train_split, cfg.data.val_split)
    log.info("Split: %d train / %d val / %d test", len(df_train), len(df_val), len(df_test))

    X_train, y_train = _to_tensors(df_train)
    X_val, y_val = _to_tensors(df_val)
    X_test, y_test = _to_tensors(df_test)

    pump_params = PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    )
    physics = PumpPhysics(pump_params)

    pinn_config = PINNConfig(
        hidden_dims=list(cfg.model.hidden_dims),
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
        lambda_data=cfg.physics.lambda_data,
        lambda_physics=cfg.physics.lambda_physics,
        lr=cfg.training.lr,
    )

    ensemble = PINNEnsemble(
        config=pinn_config,
        n_members=cfg.ensemble.n_members,
        physics=physics,
    )

    with mlflow.start_run():
        mlflow.log_params({
            "n_members": cfg.ensemble.n_members,
            "hidden_dims": str(cfg.model.hidden_dims),
            "lambda_physics": cfg.physics.lambda_physics,
            "max_epochs": cfg.training.max_epochs,
        })

        log.info("Training ensemble of %d members...", cfg.ensemble.n_members)
        ensemble.fit(
            X_train, y_train, X_val, y_val,
            max_epochs=cfg.training.max_epochs,
            batch_size=cfg.training.batch_size,
            early_stop_patience=cfg.training.early_stop_patience,
            accelerator=cfg.training.accelerator,
        )

        # Evaluate on test set
        pred = ensemble.predict(X_test)
        rmse = ((pred.mean - y_test) ** 2).mean().sqrt().item()
        mean_total_std = pred.total_std.mean().item()
        within_2sigma = (
            (y_test >= pred.mean - 2 * pred.total_std) &
            (y_test <= pred.mean + 2 * pred.total_std)
        ).float().mean().item()

        log.info("Test RMSE: %.3f m | Mean uncertainty: %.3f m | 2σ coverage: %.1f%%",
                 rmse, mean_total_std, within_2sigma * 100)

        mlflow.log_metrics({
            "test_rmse": rmse,
            "mean_total_std": mean_total_std,
            "coverage_2sigma": within_2sigma,
        })

        # Save ensemble
        out_dir = Path("models/pump_ensemble")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, member in enumerate(ensemble.members):
            torch.save(member.state_dict(), out_dir / f"member_{i:02d}.pt")
        log.info("Saved %d members to %s", len(ensemble.members), out_dir)


if __name__ == "__main__":
    main()
