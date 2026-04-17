"""Conformal prediction calibrator for ensemble surrogates.

Given a calibration set of (y, mean, total_std) triples, finds q̂ such that:
    P(|y − mean| ≤ q̂ · total_std) ≥ coverage

on future exchangeable test points. This is distribution-free and provides
a finite-sample coverage guarantee (Angelopoulos & Bates 2021, eq. 2).

Usage:
    cal = ConformalCalibrator()
    q_hat = cal.calibrate(y_cal, pred.mean, pred.total_std, coverage=0.90)
    interval = cal.predict_interval(pred.mean, pred.total_std)
    print(f"Coverage: {cal.empirical_coverage(y_test, interval):.1%}")
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CalibratedInterval:
    lower: torch.Tensor        # [N, 1]
    upper: torch.Tensor        # [N, 1]
    coverage_target: float
    q_hat: float


class ConformalCalibrator:
    """Inductive conformal prediction for heteroscedastic ensemble output."""

    def __init__(self) -> None:
        self.q_hat: float | None = None
        self._coverage_target: float = 0.90

    def calibrate(
        self,
        y_cal: torch.Tensor,
        pred_mean: torch.Tensor,
        pred_total_std: torch.Tensor,
        coverage: float = 0.90,
    ) -> float:
        """Compute q̂ from a held-out calibration set.

        Nonconformity score = |y − mean| / total_std.
        q̂ = ⌈(n+1)·coverage⌉/n quantile of scores.

        Args:
            y_cal: True values [N, 1].
            pred_mean: Ensemble mean predictions [N, 1].
            pred_total_std: Ensemble total std [N, 1].
            coverage: Target coverage level (e.g. 0.90 for 90%).

        Returns:
            q̂ — also stored in self.q_hat.
        """
        n = len(y_cal)
        scores = (
            (y_cal - pred_mean).abs() / (pred_total_std + 1e-8)
        ).squeeze().detach().numpy()
        # Finite-sample adjustment (Angelopoulos & Bates 2021)
        level = min(np.ceil((n + 1) * coverage) / n, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        self._coverage_target = coverage
        return self.q_hat

    def predict_interval(
        self,
        pred_mean: torch.Tensor,
        pred_total_std: torch.Tensor,
    ) -> CalibratedInterval:
        """Scale prediction intervals by q̂."""
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_interval()")
        half_width = self.q_hat * pred_total_std
        return CalibratedInterval(
            lower=pred_mean - half_width,
            upper=pred_mean + half_width,
            coverage_target=self._coverage_target,
            q_hat=self.q_hat,
        )

    def empirical_coverage(
        self,
        y: torch.Tensor,
        interval: CalibratedInterval,
    ) -> float:
        """Fraction of y values within [lower, upper]."""
        covered = (y >= interval.lower) & (y <= interval.upper)
        return covered.float().mean().item()
