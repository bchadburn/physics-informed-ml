"""Physics-Informed Neural Network for centrifugal pump head prediction.

Architecture:
  Input:  [flow_rate, speed, operating_hours]  (normalized)
  Hidden: N layers × width units, Tanh activation
  Output: [head_mean, head_logvar]  (heteroscedastic uncertainty)

Physics loss: enforces H ∝ N² (affinity law) via autograd.
  dH/dN should equal 2*H/N at any operating point.
  loss_physics = MSE(dH/dN, 2*H/N)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import lightning as L
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.physics_models.pump import PumpPhysics


@dataclass
class PINNConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64, 64])
    activation: str = "tanh"
    dropout: float = 0.0
    lambda_data: float = 1.0
    lambda_physics: float = 0.1
    lr: float = 1e-3


class PINN(L.LightningModule):
    """Physics-Informed Neural Network with heteroscedastic uncertainty output."""

    def __init__(
        self,
        config: PINNConfig,
        physics: "PumpPhysics | None" = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.physics = physics

        # Input normalization statistics (set during fit, defaults are safe)
        self.register_buffer("x_mean", torch.zeros(3))
        self.register_buffer("x_std", torch.ones(3))

        # Build MLP
        act = nn.Tanh() if config.activation == "tanh" else nn.ReLU()
        layers: list[nn.Module] = []
        in_dim = 3
        for h in config.hidden_dims:
            layers += [nn.Linear(in_dim, h), act]
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        # Two outputs: mean and log-variance
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (head_mean, head_logvar), each shape [N, 1]."""
        x_norm = (x - self.x_mean) / (self.x_std + 1e-8)
        out = self.net(x_norm)
        mean = out[:, :1]
        logvar = out[:, 1:]
        return mean, logvar

    def compute_losses(
        self, x: torch.Tensor, y_head: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute data loss + physics residual loss.

        x columns: [flow_rate (0), speed (1), operating_hours (2)]
        y_head: true head measurements, shape [N, 1]
        """
        mean, logvar = self(x)

        # Heteroscedastic negative log-likelihood
        # NLL = 0.5 * (logvar + (y - mean)² / exp(logvar))
        loss_data = 0.5 * (logvar + (y_head - mean) ** 2 / (logvar.exp() + 1e-8))
        loss_data = loss_data.mean()

        # Physics loss: affinity law H ∝ N²
        # d(mean)/d(speed) should equal 2 * mean / speed
        if x.requires_grad:
            grad = torch.autograd.grad(
                outputs=mean.sum(),
                inputs=x,
                create_graph=True,
            )[0]
            dH_dN = grad[:, 1:2]                     # derivative w.r.t. speed col
            speed = x[:, 1:2]
            target_dH_dN = 2.0 * mean.detach() / (speed + 1e-8)
            loss_physics = nn.functional.mse_loss(dH_dN, target_dH_dN)
        else:
            loss_physics = torch.tensor(0.0, device=x.device)

        loss_total = (
            self.config.lambda_data * loss_data
            + self.config.lambda_physics * loss_physics
        )
        return {
            "loss_data": loss_data,
            "loss_physics": loss_physics,
            "loss_total": loss_total,
        }

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        x = x.requires_grad_(True)
        losses = self.compute_losses(x, y)
        self.log("train/loss_total", losses["loss_total"], prog_bar=True)
        self.log("train/loss_data", losses["loss_data"])
        self.log("train/loss_physics", losses["loss_physics"])
        return losses["loss_total"]

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        x = x.requires_grad_(True)
        losses = self.compute_losses(x, y)
        self.log("val/loss_total", losses["loss_total"], prog_bar=True)
        self.log("val/loss_physics", losses["loss_physics"])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
