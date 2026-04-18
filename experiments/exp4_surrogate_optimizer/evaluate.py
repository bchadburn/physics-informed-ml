#!/usr/bin/env python3
"""Surrogate-in-the-loop optimization: benchmark gradient descent vs baselines.

Usage:
    uv run python experiments/exp4_surrogate_optimizer/evaluate.py
    uv run python experiments/exp4_surrogate_optimizer/evaluate.py optimization.h_min=25.0
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.benchmark import BenchmarkResult, render_markdown_table
from core.metrics import optimality_gap
from experiments.exp4_surrogate_optimizer.baselines import (
    OptimizationResult, grid_search, random_search,
)
from experiments.exp4_surrogate_optimizer.optimizer import optimize
from src.physics_models.data_generator import generate_pump_field_data
from src.physics_models.pump import PumpParameters, PumpPhysics
from src.surrogates.ensemble import PINNEnsemble
from src.surrogates.pinn import PINNConfig

log = logging.getLogger(__name__)

_PUMP_PARAMS = PumpParameters(
    design_flow=0.05,
    design_head=30.0,
    design_speed=1450.0,
    design_efficiency=0.75,
)


def _make_pinn_config(cfg: DictConfig) -> PINNConfig:
    return PINNConfig(
        hidden_dims=list(cfg.model.hidden_dims),
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
        lambda_data=cfg.model.lambda_data,
        lambda_physics=cfg.model.lambda_physics,
        lr=cfg.model.lr,
        mse_warmup_epochs=cfg.model.mse_warmup_epochs,
    )


def _load_or_train_ensemble(cfg: DictConfig) -> PINNEnsemble:
    """Load pre-trained ensemble if available, otherwise train a quick one."""
    ensemble_dir = Path(cfg.model.ensemble_dir)
    member_paths = sorted(ensemble_dir.glob("member_*.pt"))
    physics = PumpPhysics(_PUMP_PARAMS)
    pinn_config = _make_pinn_config(cfg)
    ensemble = PINNEnsemble(config=pinn_config, n_members=cfg.model.n_members, physics=physics)

    if len(member_paths) >= cfg.model.n_members:
        log.info("Loading pre-trained ensemble from %s", ensemble_dir)
        for i, member in enumerate(ensemble.members):
            member.load_state_dict(
                torch.load(member_paths[i], weights_only=True, map_location="cpu")
            )
    else:
        log.info(
            "No pre-trained ensemble found — training quick surrogate (%d epochs)...",
            cfg.model.quick_train_epochs,
        )
        df = generate_pump_field_data(n_samples=500, seed=cfg.seed)
        X = torch.tensor(
            df[["flow_rate", "speed", "operating_hours"]].values, dtype=torch.float32
        )
        y = torch.tensor(df[["head"]].values, dtype=torch.float32)
        n_train = int(len(X) * 0.8)
        ensemble.fit(
            X[:n_train], y[:n_train], X[n_train:], y[n_train:],
            max_epochs=cfg.model.quick_train_epochs,
            batch_size=64,
            early_stop_patience=20,
            accelerator="cpu",
        )

    return ensemble


def _make_surrogate_fn(ensemble: PINNEnsemble) -> Callable:
    """Differentiable surrogate: averages member means WITHOUT torch.no_grad().

    Calling without no_grad is the critical property that enables gradient-based
    optimisation — torch.autograd can differentiate through the network to inputs.
    """
    def surrogate_fn(X: torch.Tensor) -> torch.Tensor:
        preds = []
        for member in ensemble.members:
            member.eval()
            mean, _ = member(X)  # no no_grad — gradients flow
            preds.append(mean)
        return torch.stack(preds, dim=0).mean(dim=0)  # [N, 1]
    return surrogate_fn


@hydra.main(config_path="../../configs", config_name="surrogate_optimizer", version_base=None)
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    ensemble = _load_or_train_ensemble(cfg)
    surrogate_fn = _make_surrogate_fn(ensemble)

    h_min = cfg.optimization.h_min
    op_hrs = cfg.optimization.operating_hours

    log.info("Running grid search (reference, %d² evaluations)...", cfg.baselines.grid_n)
    grid_result = grid_search(surrogate_fn, h_min, op_hrs, n_grid=cfg.baselines.grid_n)
    log.info(
        "Grid: power=%.4f  head=%.2f m  violation=%.3f  calls=%d  time=%.0f ms",
        grid_result.obj_value, grid_result.head_at_opt,
        grid_result.constraint_violation, grid_result.n_surrogate_calls,
        grid_result.wall_time_ms,
    )

    log.info("Running random search (%d samples)...", cfg.baselines.random_n_samples)
    rand_result = random_search(surrogate_fn, h_min, op_hrs, n_samples=cfg.baselines.random_n_samples)
    log.info(
        "Random: power=%.4f  head=%.2f m  violation=%.3f  calls=%d  time=%.0f ms",
        rand_result.obj_value, rand_result.head_at_opt,
        rand_result.constraint_violation, rand_result.n_surrogate_calls,
        rand_result.wall_time_ms,
    )

    log.info("Running gradient optimizer (%d steps)...", cfg.optimization.n_steps)
    grad_result = optimize(
        surrogate_fn, h_min, op_hrs,
        n_steps=cfg.optimization.n_steps,
        lr=cfg.optimization.lr,
        penalty_weight=cfg.optimization.penalty_weight,
    )
    log.info(
        "Gradient: power=%.4f  head=%.2f m  violation=%.3f  calls=%d  time=%.0f ms",
        grad_result.obj_value, grad_result.head_at_opt,
        grad_result.constraint_violation, grad_result.n_surrogate_calls,
        grad_result.wall_time_ms,
    )

    gap_rand = optimality_gap(rand_result.obj_value, grid_result.obj_value)
    gap_grad = optimality_gap(grad_result.obj_value, grid_result.obj_value)
    speedup = grid_result.n_surrogate_calls / max(grad_result.n_surrogate_calls, 1)

    log.info(
        "Optimality gaps — random: %.1f%%  gradient: %.1f%%  speedup: %.0f×",
        gap_rand * 100, gap_grad * 100, speedup,
    )

    results = [
        BenchmarkResult(
            name="Grid search (reference)",
            relative_l2=0.0,
            inference_time_ms=grid_result.wall_time_ms,
            n_train_samples=grid_result.n_surrogate_calls,
            notes=f"power={grid_result.obj_value:.4f}, violation={grid_result.constraint_violation:.3f}",
        ),
        BenchmarkResult(
            name="Random search",
            relative_l2=gap_rand,
            inference_time_ms=rand_result.wall_time_ms,
            n_train_samples=rand_result.n_surrogate_calls,
            notes=f"power={rand_result.obj_value:.4f}, violation={rand_result.constraint_violation:.3f}",
        ),
        BenchmarkResult(
            name="Gradient optimizer",
            relative_l2=gap_grad,
            inference_time_ms=grad_result.wall_time_ms,
            n_train_samples=grad_result.n_surrogate_calls,
            notes=f"power={grad_result.obj_value:.4f}, violation={grad_result.constraint_violation:.3f}, {speedup:.0f}× faster",
        ),
    ]
    table = render_markdown_table(results)
    log.info("\n%s", table)

    out = Path("experiments/exp4_surrogate_optimizer/results.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Surrogate Optimizer Benchmark\n\n{table}\n")
    log.info("Saved results to %s", out)

    with mlflow.start_run():
        mlflow.log_params({
            "h_min": h_min,
            "n_steps": cfg.optimization.n_steps,
            "grid_n": cfg.baselines.grid_n,
            "random_n_samples": cfg.baselines.random_n_samples,
        })
        mlflow.log_metrics({
            "grid_power": grid_result.obj_value,
            "random_power": rand_result.obj_value,
            "grad_power": grad_result.obj_value,
            "optimality_gap_random": gap_rand,
            "optimality_gap_gradient": gap_grad,
            "gradient_speedup_vs_grid": speedup,
            "grad_constraint_violation": grad_result.constraint_violation,
        })


if __name__ == "__main__":
    main()
