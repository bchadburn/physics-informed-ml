#!/usr/bin/env python3
"""Evaluate trained FNO2d and UNet2d on Darcy flow.

Usage:
    uv run python experiments/exp2_fno_heat_transfer/evaluate.py
    uv run python experiments/exp2_fno_heat_transfer/evaluate.py model.name=unet2d
"""
import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.benchmark import BenchmarkResult, render_markdown_table
from core.metrics import inference_timer, relative_l2_error
from experiments.exp2_fno_heat_transfer.data import DarcyDataset, load_or_generate
from experiments.exp2_fno_heat_transfer.model import FNO2d, UNet2d
from experiments.exp2_fno_heat_transfer.train import HeatTransferModule

log = logging.getLogger(__name__)


def _quick_train(module, train_ds, val_ds, batch_size, max_epochs=30, accelerator="cpu"):
    """Train for a small number of epochs to get a usable model."""
    import lightning as pl
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(
        module,
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def _eval_rel_l2(module, dataset, batch_size=32):
    """Compute mean relative L2 error on dataset."""
    loader = DataLoader(dataset, batch_size=batch_size)
    device = next(module.parameters()).device
    preds, targets = [], []
    module.eval()
    with torch.no_grad():
        for x, y in loader:
            preds.append(module(x.to(device)).cpu())
            targets.append(y.cpu())
    pred = torch.cat(preds)
    true = torch.cat(targets)
    return relative_l2_error(pred.numpy(), true.numpy())


@hydra.main(config_path="../../configs", config_name="fno_heat", version_base=None)
def main(cfg: DictConfig) -> None:
    # --- build training dataset ---
    kappa, T = load_or_generate(cfg.data.cache_path, cfg.data.n_samples, cfg.data.grid_size, cfg.seed)
    ds = DarcyDataset(kappa, T)
    n = len(ds)
    n_train = int(n * cfg.data.train_frac)
    n_val = int(n * cfg.data.val_frac)
    train_ds = Subset(ds, range(n_train))
    val_ds = Subset(ds, range(n_train, n_train + n_val))
    test_ds = Subset(ds, range(n_train + n_val, n))

    # --- build and quick-train model ---
    if cfg.model.name == "fno2d":
        model = FNO2d(cfg.model.modes1, cfg.model.modes2, cfg.model.width, cfg.model.n_layers)
    else:
        model = UNet2d(cfg.model.width)

    module = HeatTransferModule(model, lr=cfg.training.lr)
    log.info("Quick-training %s for 30 epochs...", cfg.model.name)
    _quick_train(module, train_ds, val_ds, cfg.training.batch_size, max_epochs=30,
                 accelerator=cfg.training.accelerator)

    # --- in-distribution test ---
    test_rel_l2 = _eval_rel_l2(module, test_ds)
    log.info("In-dist test rel-L2: %.4f", test_rel_l2)

    # --- OOD: kappa up to 24 (2x train max) ---
    ood_cache = Path(cfg.data.cache_path).parent / f"darcy_{cfg.data.grid_size}_ood.h5"
    kappa_ood, T_ood = load_or_generate(str(ood_cache), 200, cfg.data.grid_size, seed=99, max_kappa=24.0)
    ood_ds = DarcyDataset(kappa_ood, T_ood)
    ood_rel_l2 = _eval_rel_l2(module, ood_ds)
    log.info("OOD (kappa<=24) rel-L2: %.4f", ood_rel_l2)

    # --- resolution generalization: 128x128 zero-shot ---
    res_cache = Path(cfg.data.cache_path).parent / "darcy_128.h5"
    kappa_res, T_res = load_or_generate(str(res_cache), 100, 128, seed=42)
    res_ds = DarcyDataset(kappa_res, T_res)
    res_rel_l2 = _eval_rel_l2(module, res_ds)
    log.info("Resolution 128x128 rel-L2: %.4f", res_rel_l2)

    # --- inference timing ---
    sample_input = torch.randn(1, 3, cfg.data.grid_size, cfg.data.grid_size)
    t_ms = inference_timer(module, sample_input, n_runs=50, device="cpu") * 1000

    # --- benchmark table ---
    results = [
        BenchmarkResult(
            name=f"{cfg.model.name} (in-dist)",
            relative_l2=test_rel_l2,
            inference_time_ms=t_ms,
            n_train_samples=n_train,
        ),
        BenchmarkResult(
            name=f"{cfg.model.name} (OOD kappa<=24)",
            relative_l2=ood_rel_l2,
            inference_time_ms=t_ms,
            n_train_samples=n_train,
            notes="Zero-shot, kappa 2x train max",
        ),
        BenchmarkResult(
            name=f"{cfg.model.name} (128x128 zero-shot)",
            relative_l2=res_rel_l2,
            inference_time_ms=t_ms,
            n_train_samples=n_train,
            notes="Zero-shot resolution generalization",
        ),
    ]
    table = render_markdown_table(results)
    log.info("\n%s", table)

    out = Path("experiments/exp2_fno_heat_transfer/results.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Darcy Flow Benchmark\n\n{table}\n")
    log.info("Saved benchmark to %s", out)


if __name__ == "__main__":
    main()
