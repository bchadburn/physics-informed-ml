"""Bayesian linear regression on vendor-curve residual.

Math (Bishop 2006, Ch 3.3):
    Prior:      w ~ N(0, α⁻¹ I)
    Likelihood: y | w, x ~ N(φ(x)ᵀ w, β⁻¹)
    Posterior:  w | data ~ N(m_N, S_N)

        S_N⁻¹ = α I + β Φᵀ Φ
        m_N   = β S_N Φᵀ y

    Predictive: p(y* | x*) = N(m_Nᵀ φ*, φ*ᵀ S_N φ* + β⁻¹)

Sequential rank-1 update (new single observation φ_new, y_new):
    S_N_new⁻¹ = S_N⁻¹ + β φ_new φ_newᵀ
    m_N_new   = S_N_new (S_N⁻¹ m_N + β φ_new y_new)
"""
from __future__ import annotations

import numpy as np


class BayesianResidualModel:
    """Bayesian linear regression on the residual η_true − η_vendor.

    Args:
        alpha: prior precision (higher = stronger pull toward zero weights)
        beta:  noise precision = 1 / σ_noise²
    """

    def __init__(self, alpha: float = 1.0, beta: float = 40000.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self._m: np.ndarray | None = None
        self._S: np.ndarray | None = None

    def fit_batch(self, phi: np.ndarray, y_residual: np.ndarray) -> None:
        """Full batch posterior update.

        Args:
            phi:        design matrix, shape (n_samples, n_features)
            y_residual: residuals η_true − η_vendor, shape (n_samples,)
        """
        n_features = phi.shape[1]
        S0_inv = self.alpha * np.eye(n_features)
        S_inv = S0_inv + self.beta * phi.T @ phi
        self._S = np.linalg.inv(S_inv)
        self._m = self.beta * self._S @ phi.T @ y_residual

    def update(self, phi_new: np.ndarray, y_new: float) -> None:
        """Sequential rank-1 posterior update for a single new observation.

        Args:
            phi_new: feature vector, shape (n_features,)
            y_new:   observed residual scalar
        """
        if self._S is None:
            raise RuntimeError("Call fit_batch() before update()")
        phi_new = np.asarray(phi_new).reshape(-1)
        S_inv = np.linalg.inv(self._S)
        S_inv_new = S_inv + self.beta * np.outer(phi_new, phi_new)
        self._S = np.linalg.inv(S_inv_new)
        self._m = self._S @ (S_inv @ self._m + self.beta * phi_new * y_new)

    def predict(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and std for each row of phi.

        Args:
            phi: design matrix, shape (n_samples, n_features)

        Returns:
            (mean, std) each shape (n_samples,)
        """
        if self._m is None:
            raise RuntimeError("Call fit_batch() before predict()")
        phi = np.atleast_2d(phi)
        mean = phi @ self._m
        var = np.array([p @ self._S @ p for p in phi]) + 1.0 / self.beta
        return mean, np.sqrt(var)
