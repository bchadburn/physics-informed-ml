"""Tests for evaluate.py helper functions."""
import torch


def test_eval_rel_l2_shape(tmp_path):
    """_eval_rel_l2 returns a scalar float."""
    from experiments.exp2_fno_heat_transfer.data import load_or_generate, DarcyDataset
    from experiments.exp2_fno_heat_transfer.model import FNO2d
    from experiments.exp2_fno_heat_transfer.train import HeatTransferModule
    from experiments.exp2_fno_heat_transfer.evaluate import _eval_rel_l2

    kappa, T = load_or_generate(tmp_path / "d.h5", 10, 16, seed=0)
    ds = DarcyDataset(kappa, T)
    model = FNO2d(modes1=4, modes2=4, width=8, n_layers=2)
    module = HeatTransferModule(model, lr=1e-3)
    result = _eval_rel_l2(module, ds)
    assert isinstance(result, float)
    assert result > 0
