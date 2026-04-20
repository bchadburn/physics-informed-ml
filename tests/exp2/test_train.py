"""Smoke test for HeatTransferModule training."""
import lightning as pl
import torch
from torch.utils.data import DataLoader

from experiments.exp2_fno_heat_transfer.data import DarcyDataset, load_or_generate
from experiments.exp2_fno_heat_transfer.model import FNO2d
from experiments.exp2_fno_heat_transfer.train import HeatTransferModule


def test_training_runs_one_epoch(tmp_path):
    kappa, T = load_or_generate(tmp_path / "darcy.h5", n_samples=20, grid_size=16, seed=0)
    ds = DarcyDataset(kappa, T)
    train_ds = torch.utils.data.Subset(ds, range(16))
    val_ds = torch.utils.data.Subset(ds, range(16, 20))
    model = FNO2d(modes1=4, modes2=4, width=8, n_layers=2)
    module = HeatTransferModule(model, lr=1e-3)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False, logger=False)
    trainer.fit(module, DataLoader(train_ds, batch_size=4), DataLoader(val_ds, batch_size=4))
    assert trainer.current_epoch == 1
