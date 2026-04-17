"""FNO2d and UNet2d models for 2D steady-state heat transfer.

FNO2d (Li et al. 2021) learns the solution operator κ → T:
  - SpectralConv2d: FFT → element-wise complex multiply → IFFT (low-freq only)
  - Each FNO block: SpectralConv2d + pointwise Conv2d, outputs added
  - Lifting: 3 input channels → width; Projection: width → 128 → 1
  - Resolution-invariant: same weights work at any grid size >= 2*modes

UNet2d is a standard encoder-decoder baseline. It can process any resolution
but does NOT generalise zero-shot to new resolutions in the same way FNO does,
because it has learned spatial scale-specific features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Fourier integral operator layer.

    Computes: (W · F[x])_low + (skip) where W are learned complex weights.
    Only the lowest `modes1 × modes2` Fourier coefficients are modified;
    higher frequencies pass through as zero (implicit low-pass filter).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        # Two sets of weights: lower-left and lower-right Fourier quadrants
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(
        x: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """Batched complex matrix multiply: (B,in,H,W),(in,out,H,W) → (B,out,H,W)."""
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)  # (B, C, H, W//2+1) complex

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        # Lower-left quadrant
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Lower-right quadrant (negative frequencies in first dim)
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(H, W))  # (B, out_channels, H, W)


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D scalar fields.

    Input:  (B, 3, H, W) — [κ, x_coord, y_coord]
    Output: (B, 1, H, W) — temperature field T

    Architecture:
        fc0 (lift):  Linear 3 → width  (applied pointwise via permute trick)
        × n_layers:  SpectralConv2d(width→width) + Conv2d 1×1(width→width), sum, GELU
        fc1, fc2:    Linear width → 128 → 1  (applied pointwise)

    Resolution-invariant: spectral modes are low-frequency components that
    exist at any grid resolution. Same weights run at 32×32, 64×64, 128×128.
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(3, width)

        self.convs = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)
        ])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        # Lift channels: permute so Linear acts on channel dim
        x = x.permute(0, 2, 3, 1)   # (B, H, W, 3)
        x = self.fc0(x)               # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)    # (B, width, H, W)

        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))

        x = x.permute(0, 2, 3, 1)    # (B, H, W, width)
        x = F.gelu(self.fc1(x))       # (B, H, W, 128)
        x = self.fc2(x)               # (B, H, W, 1)
        return x.permute(0, 3, 1, 2)  # (B, 1, H, W)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2d(nn.Module):
    """Baseline UNet for comparison with FNO2d.

    Roughly matched parameter count to FNO2d (width=32).
    3-level encoder/decoder with skip connections.

    Input:  (B, 3, H, W) — same format as FNO2d
    Output: (B, 1, H, W) — temperature field T

    Note: unlike FNO2d, UNet2d is NOT resolution-invariant. Features learned
    at 64×64 encode spatial scale information that does not transfer to 128×128.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        w = width
        self.enc1 = _DoubleConv(3, w)
        self.enc2 = _DoubleConv(w, w * 2)
        self.enc3 = _DoubleConv(w * 2, w * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _DoubleConv(w * 4, w * 8)

        self.up1 = nn.ConvTranspose2d(w * 8, w * 4, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(w * 8, w * 4)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(w * 4, w * 2)
        self.up3 = nn.ConvTranspose2d(w * 2, w, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(w * 2, w)

        self.out = nn.Conv2d(w, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d1 = self.dec1(torch.cat([self.up1(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], dim=1))
        return self.out(d3)
