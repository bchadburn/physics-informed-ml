"""Synthetic 2D Darcy flow dataset generator.

Generates (κ, T) pairs where:
  κ(x,y) — random smooth conductivity field in [1, max_kappa]
  T(x,y) — solution to −∇·(κ∇T) = 1 with T=0 on boundary

Solved via finite differences using scipy's sparse direct solver.
Dataset is cached to HDF5 to avoid re-solving on every run.
"""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def _random_kappa(grid_size: int, rng: np.random.Generator, max_kappa: float = 12.0) -> np.ndarray:
    """Smooth random conductivity field in [1, max_kappa] via truncated Fourier modes."""
    N = grid_size
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    field = np.zeros((N, N))
    for k1 in range(1, 6):
        for k2 in range(1, 6):
            amp = rng.standard_normal() / (k1 ** 2 + k2 ** 2)
            phase = rng.uniform(0, 2 * np.pi)
            field += amp * np.cos(k1 * np.pi * X + k2 * np.pi * Y + phase)

    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return 1.0 + (max_kappa - 1.0) * field


def _build_darcy_matrix(kappa: np.ndarray) -> sparse.csr_matrix:
    """Build FD matrix for interior points of −∇·(κ∇u) = f, u=0 on boundary.

    Uses arithmetic-mean interface conductivities (standard for heterogeneous κ).
    N = full grid size; M = N-2 interior points per dimension.
    """
    N = kappa.shape[0]
    M = N - 2
    h = 1.0 / (N - 1)

    # Interface conductivities on interior grid (M × M), using full-grid κ
    k_e = 0.5 * (kappa[1:N-1, 1:N-1] + kappa[1:N-1, 2:N  ])  # east
    k_w = 0.5 * (kappa[1:N-1, 0:N-2] + kappa[1:N-1, 1:N-1])  # west
    k_n = 0.5 * (kappa[1:N-1, 1:N-1] + kappa[2:N,   1:N-1])  # north
    k_s = 0.5 * (kappa[0:N-2, 1:N-1] + kappa[1:N-1, 1:N-1])  # south

    diag = (k_e + k_w + k_n + k_s) / h ** 2  # (M, M)

    idx = np.arange(M * M).reshape(M, M)

    r, c, v = [], [], []

    # Diagonal
    r.append(idx.ravel()); c.append(idx.ravel()); v.append(diag.ravel())
    # East
    r.append(idx[:, :-1].ravel()); c.append(idx[:, 1:].ravel())
    v.append((-k_e[:, :-1] / h ** 2).ravel())
    # West
    r.append(idx[:, 1:].ravel()); c.append(idx[:, :-1].ravel())
    v.append((-k_w[:, 1:] / h ** 2).ravel())
    # North
    r.append(idx[:-1, :].ravel()); c.append(idx[1:, :].ravel())
    v.append((-k_n[:-1, :] / h ** 2).ravel())
    # South
    r.append(idx[1:, :].ravel()); c.append(idx[:-1, :].ravel())
    v.append((-k_s[1:, :] / h ** 2).ravel())

    rows = np.concatenate(r)
    cols = np.concatenate(c)
    vals = np.concatenate(v)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(M * M, M * M)).tocsr()


def _solve_darcy(kappa: np.ndarray) -> np.ndarray:
    """Solve −∇·(κ∇T) = 1 with T=0 on boundary. Returns T on full N×N grid."""
    N = kappa.shape[0]
    M = N - 2
    A = _build_darcy_matrix(kappa)
    b = np.ones(M * M)
    T_int = spsolve(A, b)
    T = np.zeros((N, N))
    T[1:N-1, 1:N-1] = T_int.reshape(M, M)
    return T


def generate_darcy_dataset(
    n_samples: int,
    grid_size: int = 64,
    seed: int = 42,
    max_kappa: float = 12.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n_samples (κ, T) pairs on a grid_size × grid_size grid.

    Args:
        n_samples: Number of PDE solutions to generate.
        grid_size: Full grid dimension including boundary (e.g. 64).
        seed: Random seed for κ field generation.
        max_kappa: Maximum conductivity value (range is [1, max_kappa]).
        verbose: Print progress every 100 samples.

    Returns:
        kappa: (n_samples, grid_size, grid_size) conductivity fields.
        T: (n_samples, grid_size, grid_size) temperature solutions.
    """
    rng = np.random.default_rng(seed)
    kappa_arr = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    T_arr = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)

    for i in range(n_samples):
        kappa = _random_kappa(grid_size, rng, max_kappa=max_kappa)
        T = _solve_darcy(kappa)
        kappa_arr[i] = kappa.astype(np.float32)
        T_arr[i] = T.astype(np.float32)
        if verbose and (i + 1) % 100 == 0:
            log.info("Generated %d/%d samples", i + 1, n_samples)

    return kappa_arr, T_arr


def load_or_generate(
    cache_path: Path,
    n_samples: int,
    grid_size: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset from HDF5 cache if it exists, otherwise generate and save."""
    if cache_path.exists():
        with h5py.File(cache_path, "r") as f:
            cached_kappa = f["kappa"][:]
        if cached_kappa.shape != (n_samples, grid_size, grid_size):
            cache_path.unlink()
        else:
            with h5py.File(cache_path, "r") as f:
                return f["kappa"][:], f["T"][:]
    kappa, T = generate_darcy_dataset(n_samples, grid_size, seed, verbose=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "w") as f:
        f.create_dataset("kappa", data=kappa, compression="gzip")
        f.create_dataset("T", data=T, compression="gzip")
    return kappa, T


def get_grid_coords(grid_size: int) -> torch.Tensor:
    """Return (x, y) coordinate grids for a grid_size × grid_size domain.

    Returns:
        Tensor of shape (2, grid_size, grid_size).
        [0] = x coordinates (column), [1] = y coordinates (row), both in [0, 1].
    """
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)


class DarcyDataset(Dataset):
    """PyTorch Dataset wrapping (κ, T) arrays.

    Each item is (x, y) where:
        x: (3, H, W) tensor — [κ, x_coord, y_coord]
        y: (1, H, W) tensor — temperature field T
    """

    def __init__(self, kappa: np.ndarray, T: np.ndarray) -> None:
        self.kappa = torch.from_numpy(kappa).float().unsqueeze(1)  # (N, 1, H, W)
        self.T = torch.from_numpy(T).float().unsqueeze(1)           # (N, 1, H, W)
        H, W = kappa.shape[1], kappa.shape[2]
        coords = get_grid_coords(H)                                  # (2, H, W)
        # Broadcast coords to all samples
        self._coords = coords.unsqueeze(0).expand(len(kappa), -1, -1, -1)  # (N, 2, H, W)

    def __len__(self) -> int:
        return len(self.kappa)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self.kappa[idx], self._coords[idx]], dim=0)  # (3, H, W)
        return x, self.T[idx]                                         # (3,H,W), (1,H,W)
