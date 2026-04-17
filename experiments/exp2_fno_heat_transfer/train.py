"""FNO training with PyTorch Lightning + Hydra for 2D Darcy heat transfer."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.metrics import relative_l2_error
from experiments.exp2_fno_heat_transfer.data import DarcyDataset, load_or_generate
from experiments.exp2_fno_heat_transfer.model import FNO2d, UNet2d

log = logging.getLogger(__name__)


class HeatTransferModule(pl.LightningModule):
    """Lightning module wrapping FNO2d or UNet2d for heat transfer prediction."""

    def __init__(self, model: nn.Module, lr: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], split: str) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)

        pred_np = pred.detach().cpu().numpy()
        true_np = y.detach().cpu().numpy()
        rel_l2 = relative_l2_error(pred_np, true_np)

        self.log(f"{split}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{split}/rel_l2", rel_l2, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@hydra.main(config_path="../../configs", config_name="fno_heat", version_base=None)
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    cache_path = Path(cfg.data.cache_path)
    kappa, T = load_or_generate(
        cache_path,
        n_samples=cfg.data.n_samples,
        grid_size=cfg.data.grid_size,
        seed=cfg.seed,
    )

    ds = DarcyDataset(kappa, T)
    n = len(ds)
    n_train = int(n * cfg.data.train_frac)
    n_val = int(n * cfg.data.val_frac)

    train_ds = torch.utils.data.Subset(ds, range(n_train))
    val_ds = torch.utils.data.Subset(ds, range(n_train, n_train + n_val))
    test_ds = torch.utils.data.Subset(ds, range(n_train + n_val, n))

    log.info("Split: %d train / %d val / %d test", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    if cfg.model.name == "fno2d":
        model = FNO2d(
            modes1=cfg.model.modes1,
            modes2=cfg.model.modes2,
            width=cfg.model.width,
            n_layers=cfg.model.n_layers,
        )
    else:
        model = UNet2d(width=cfg.model.width)

    module = HeatTransferModule(model, lr=cfg.training.lr)

    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=cfg.training.early_stop_patience,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        callbacks=[early_stop],
        enable_progress_bar=False,
    )

    with mlflow.start_run():
        mlflow.log_params({
            "model_name": cfg.model.name,
            "modes1": cfg.model.modes1,
            "modes2": cfg.model.modes2,
            "width": cfg.model.width,
            "n_layers": cfg.model.n_layers,
            "lr": cfg.training.lr,
            "batch_size": cfg.training.batch_size,
            "max_epochs": cfg.training.max_epochs,
            "n_samples": cfg.data.n_samples,
            "grid_size": cfg.data.grid_size,
        })

        trainer.fit(module, train_loader, val_loader)

        # Evaluate on test set
        all_preds, all_trues = [], []
        module.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                pred = module(x_batch)
                all_preds.append(pred.cpu().numpy())
                all_trues.append(y_batch.cpu().numpy())

        preds = np.concatenate(all_preds)
        trues = np.concatenate(all_trues)
        test_rel_l2 = relative_l2_error(preds, trues)

        log.info("Test relative L2 error: %.4f", test_rel_l2)
        mlflow.log_metric("test_rel_l2", test_rel_l2)


if __name__ == "__main__":
    main()
