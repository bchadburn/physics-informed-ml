"""Shared evaluation metrics for all experiments."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


def relative_l2_error(pred: np.ndarray, true: np.ndarray) -> float:
    """||pred - true||_2 / ||true||_2 — standard metric in PDE surrogate literature."""
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-8))


def expected_calibration_error(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE for regression: measures gap between predicted and empirical coverage.

    Bins the unit interval [0,1] into n_bins confidence levels. For each bin,
    checks whether the empirical fraction of true values within the predicted
    interval matches the nominal confidence level.
    """
    z = (true - pred_mean) / (pred_std + 1e-8)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        nominal = 0.5 * (lo + hi)
        z_threshold = stats.norm.ppf((1 + nominal) / 2)
        empirical = float(np.mean(np.abs(z) <= z_threshold))
        ece += abs(nominal - empirical) / n_bins
    return ece


def inference_timer(
    model: nn.Module,
    inputs: torch.Tensor,
    n_runs: int = 100,
    device: str = "cpu",
) -> float:
    """Mean wall-clock seconds per forward pass over n_runs timed calls.

    Runs 10 warm-up passes first (not timed) to avoid JIT/cache effects.
    """
    model = model.to(device).eval()
    inputs = inputs.to(device)
    with torch.no_grad():
        for _ in range(10):
            model(inputs)
    if device != "cpu":
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(inputs)
            if device != "cpu":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def optimality_gap(surrogate_obj: float, reference_obj: float) -> float:
    """Relative gap between surrogate optimum and reference (grid search) optimum.

    Positive means the surrogate found a worse solution than the reference.
    Zero means the surrogate matched the reference exactly.

    Args:
        surrogate_obj:  objective value found by surrogate-based optimizer
        reference_obj:  objective value from exhaustive reference (grid search)

    Returns:
        (surrogate_obj - reference_obj) / abs(reference_obj)
    """
    return (surrogate_obj - reference_obj) / abs(reference_obj)
