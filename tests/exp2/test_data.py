import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from experiments.exp2_fno_heat_transfer.data import (
    DarcyDataset,
    generate_darcy_dataset,
    get_grid_coords,
)


def test_generate_shapes():
    kappa, T = generate_darcy_dataset(n_samples=4, grid_size=16, seed=0)
    assert kappa.shape == (4, 16, 16)
    assert T.shape == (4, 16, 16)


def test_kappa_positive():
    kappa, _ = generate_darcy_dataset(n_samples=4, grid_size=16, seed=0)
    assert (kappa > 0).all()


def test_boundary_conditions():
    """T must be zero on all four edges (Dirichlet BCs)."""
    _, T = generate_darcy_dataset(n_samples=3, grid_size=16, seed=1)
    assert np.allclose(T[:, 0, :], 0.0, atol=1e-6)
    assert np.allclose(T[:, -1, :], 0.0, atol=1e-6)
    assert np.allclose(T[:, :, 0], 0.0, atol=1e-6)
    assert np.allclose(T[:, :, -1], 0.0, atol=1e-6)


def test_interior_positive():
    """With unit source and zero BCs, interior T must be positive."""
    _, T = generate_darcy_dataset(n_samples=3, grid_size=16, seed=2)
    assert (T[:, 1:-1, 1:-1] > 0).all()


def test_darcy_dataset_len_and_item():
    kappa, T = generate_darcy_dataset(n_samples=8, grid_size=16, seed=0)
    ds = DarcyDataset(kappa, T)
    assert len(ds) == 8
    x, y = ds[0]
    # x: (3, 16, 16) — kappa + grid_x + grid_y; y: (1, 16, 16)
    assert x.shape == (3, 16, 16)
    assert y.shape == (1, 16, 16)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_get_grid_coords_shape():
    coords = get_grid_coords(32)
    assert coords.shape == (2, 32, 32)
    # x coords span [0, 1]
    assert abs(coords[0, 0, 0].item()) < 1e-6
    assert abs(coords[0, 0, -1].item() - 1.0) < 1e-6
