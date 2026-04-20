#!/usr/bin/env python3
"""Surrogate-in-the-loop optimization: benchmark gradient descent vs baselines.

Usage:
    uv run python experiments/exp4_surrogate_optimizer/evaluate.py
    uv run python experiments/exp4_surrogate_optimizer/evaluate.py optimization.h_min=25.0
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.metrics import optimality_gap
from experiments.exp4_surrogate_optimizer.baselines import (
    OptimizationResult,
    grid_search,
    random_search,
)
from experiments.exp4_surrogate_optimizer.optimizer import optimize, optimize_multistart
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


def _render_opt_table(
    rows: list[tuple[str, OptimizationResult, float, float | None]],
) -> str:
    """Render optimizer comparison as a markdown table.

    Columns: Method | Power (W) | Head (m) | Violation | Gap (%) | Calls | Time (ms) | Speedup
    """
    header = "| Method | Power | Head (m) | Violation | Gap (%) | Calls | Time (ms) | Speedup |"
    sep    = "|--------|-------|----------|-----------|---------|-------|-----------|---------|"
    lines = [header, sep]
    for name, r, gap, speedup in rows:
        sp_str = f"{speedup:.1f}×" if speedup is not None else "—"
        lines.append(
            f"| {name} | {r.obj_value:.4f} | {r.head_at_opt:.2f} | "
            f"{r.constraint_violation:.3f} | {gap * 100:.1f} | "
            f"{r.n_surrogate_calls} | {r.wall_time_ms:.0f} | {sp_str} |"
        )
    return "\n".join(lines)


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

    log.info("Running gradient optimizer (%d steps, single start)...", cfg.optimization.n_steps)
    grad_result = optimize(
        surrogate_fn, h_min, op_hrs,
        n_steps=cfg.optimization.n_steps,
        lr=cfg.optimization.lr,
        penalty_weight=cfg.optimization.penalty_weight,
    )
    log.info(
        "Gradient (1 start): power=%.4f  head=%.2f m  violation=%.3f  calls=%d  time=%.0f ms",
        grad_result.obj_value, grad_result.head_at_opt,
        grad_result.constraint_violation, grad_result.n_surrogate_calls,
        grad_result.wall_time_ms,
    )

    log.info(
        "Running multi-start gradient optimizer (%d starts × %d steps)...",
        cfg.optimization.n_starts, cfg.optimization.n_steps,
    )
    multistart_result = optimize_multistart(
        surrogate_fn, h_min, op_hrs,
        n_starts=cfg.optimization.n_starts,
        n_steps=cfg.optimization.n_steps,
        lr=cfg.optimization.lr,
        penalty_weight=cfg.optimization.penalty_weight,
    )
    log.info(
        "Gradient (%d starts): power=%.4f  head=%.2f m  violation=%.3f  calls=%d  time=%.0f ms",
        cfg.optimization.n_starts,
        multistart_result.obj_value, multistart_result.head_at_opt,
        multistart_result.constraint_violation, multistart_result.n_surrogate_calls,
        multistart_result.wall_time_ms,
    )

    gap_rand = optimality_gap(rand_result.obj_value, grid_result.obj_value)
    gap_grad = optimality_gap(grad_result.obj_value, grid_result.obj_value)
    gap_multi = optimality_gap(multistart_result.obj_value, grid_result.obj_value)
    speedup_grad = grid_result.n_surrogate_calls / max(grad_result.n_surrogate_calls, 1)
    speedup_multi = grid_result.n_surrogate_calls / max(multistart_result.n_surrogate_calls, 1)

    log.info(
        "Optimality gaps — random: %.1f%%  gradient: %.1f%%  multi-start: %.1f%%",
        gap_rand * 100, gap_grad * 100, gap_multi * 100,
    )

    table = _render_opt_table([
        ("Grid search (reference)", grid_result, 0.0, None),
        ("Random search", rand_result, gap_rand, None),
        ("Gradient (1 start)", grad_result, gap_grad, speedup_grad),
        (f"Gradient ({cfg.optimization.n_starts} starts)", multistart_result, gap_multi, speedup_multi),
    ])
    log.info("\n%s", table)

    out = Path("experiments/exp4_surrogate_optimizer/results.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Surrogate Optimizer Benchmark\n\n{table}\n")
    log.info("Saved results to %s", out)

    with mlflow.start_run():
        mlflow.log_params({
            "h_min": h_min,
            "n_steps": cfg.optimization.n_steps,
            "n_starts": cfg.optimization.n_starts,
            "grid_n": cfg.baselines.grid_n,
            "random_n_samples": cfg.baselines.random_n_samples,
        })
        mlflow.log_metrics({
            "grid_power": grid_result.obj_value,
            "random_power": rand_result.obj_value,
            "grad_power": grad_result.obj_value,
            "multistart_power": multistart_result.obj_value,
            "optimality_gap_random": gap_rand,
            "optimality_gap_gradient": gap_grad,
            "optimality_gap_multistart": gap_multi,
            "grad_constraint_violation": grad_result.constraint_violation,
            "multistart_constraint_violation": multistart_result.constraint_violation,
        })


if __name__ == "__main__":
    main()
