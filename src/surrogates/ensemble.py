"""Ensemble of PINNs for epistemic + aleatoric uncertainty decomposition.

Epistemic uncertainty: disagreement across ensemble members (model uncertainty)
Aleatoric uncertainty: average learned logvar (irreducible data noise)
Total uncertainty:    sqrt(epistemic² + aleatoric²)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

from src.surrogates.pinn import PINN, PINNConfig

if TYPE_CHECKING:
    from src.physics_models.pump import PumpPhysics


@dataclass
class EnsemblePrediction:
    mean: torch.Tensor           # [N, 1] — average prediction across members
    epistemic_std: torch.Tensor  # [N, 1] — std of member means
    aleatoric_std: torch.Tensor  # [N, 1] — sqrt of mean(exp(logvar))
    total_std: torch.Tensor      # [N, 1] — sqrt(epistemic² + aleatoric²)


class PINNEnsemble:
    """Train and predict with an ensemble of PINNs.

    Each member is trained with a different random seed, producing diverse
    predictions whose variance estimates epistemic uncertainty.
    """

    def __init__(
        self,
        config: PINNConfig,
        n_members: int = 10,
        physics: PumpPhysics | None = None,
    ) -> None:
        self.config = config
        self.n_members = n_members
        self.physics = physics
        self.members: list[PINN] = [
            PINN(config, physics=physics) for _ in range(n_members)
        ]

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        max_epochs: int = 300,
        batch_size: int = 64,
        early_stop_patience: int = 30,
        accelerator: str = "auto",
    ) -> None:
        """Train all ensemble members. Each gets a different random seed."""
        self._x_mean = X_train.mean(dim=0)
        self._x_std = X_train.std(dim=0) + 1e-8

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_ds = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_ds, batch_size=batch_size)

        trained_members = []
        for seed in range(self.n_members):
            L.seed_everything(seed, workers=True)
            member = PINN(self.config, physics=self.physics)
            member.x_mean = self._x_mean
            member.x_std = self._x_std

            callbacks = [
                EarlyStopping(
                    monitor="val/loss_total" if val_loader else "train/loss_total",
                    patience=early_stop_patience,
                    mode="min",
                )
            ]
            trainer = L.Trainer(
                max_epochs=max_epochs,
                accelerator=accelerator,
                callbacks=callbacks,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(member, train_loader, val_loader)
            trained_members.append(member)
        self.members = trained_members

    def predict(self, X: torch.Tensor) -> EnsemblePrediction:
        """Run all members and return decomposed uncertainty.

        Args:
            X: Input tensor [N, 3] — [flow_rate, speed, operating_hours]

        Returns:
            EnsemblePrediction with mean, epistemic_std, aleatoric_std, total_std.
        """
        means = []
        vars_aleatoric = []

        for member in self.members:
            member.eval()
            with torch.no_grad():
                mean, logvar = member(X)
            means.append(mean)
            vars_aleatoric.append(logvar.exp())

        means_stack = torch.stack(means, dim=0)          # [M, N, 1]
        vars_stack = torch.stack(vars_aleatoric, dim=0)  # [M, N, 1]

        ensemble_mean = means_stack.mean(dim=0)           # [N, 1]
        epistemic_var = means_stack.var(dim=0)            # [N, 1]
        aleatoric_var = vars_stack.mean(dim=0)            # [N, 1]
        total_var = epistemic_var + aleatoric_var

        return EnsemblePrediction(
            mean=ensemble_mean,
            epistemic_std=epistemic_var.sqrt(),
            aleatoric_std=aleatoric_var.sqrt(),
            total_std=total_var.sqrt(),
        )
