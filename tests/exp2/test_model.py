import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from experiments.exp2_fno_heat_transfer.model import SpectralConv2d, FNO2d, UNet2d


def test_spectral_conv_output_shape():
    conv = SpectralConv2d(in_channels=4, out_channels=4, modes1=6, modes2=6)
    x = torch.randn(2, 4, 32, 32)
    out = conv(x)
    assert out.shape == (2, 4, 32, 32)


def test_spectral_conv_complex_weights():
    conv = SpectralConv2d(in_channels=4, out_channels=8, modes1=6, modes2=6)
    assert conv.weights1.dtype == torch.cfloat
    assert conv.weights1.shape == (4, 8, 6, 6)
    assert conv.weights2.shape == (4, 8, 6, 6)


def test_fno2d_output_shape():
    model = FNO2d(modes1=8, modes2=8, width=16, n_layers=2)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 1, 32, 32)


def test_fno2d_resolution_invariance():
    """FNO must run on any resolution >= 2*modes without retraining."""
    import torch.nn.functional as F
    model = FNO2d(modes1=8, modes2=8, width=16, n_layers=2)
    model.eval()
    def make_input(n):
        coords = torch.linspace(0, 1, n)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        kappa = 1.0 + torch.sin(2 * 3.14159 * gx) * torch.cos(2 * 3.14159 * gy)
        return torch.stack([kappa, gx, gy]).unsqueeze(0)
    with torch.no_grad():
        out_32 = model(make_input(32))
        out_64 = model(make_input(64))
        out_64_ds = F.interpolate(out_64, size=32, mode="bilinear", align_corners=False)
    corr = torch.corrcoef(torch.stack([out_32.flatten(), out_64_ds.flatten()]))[0, 1]
    assert corr > 0.5


def test_unet2d_output_shape():
    model = UNet2d(width=16)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_fno2d_gradients_flow():
    model = FNO2d(modes1=8, modes2=8, width=16, n_layers=2)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    loss = out.mean()
    loss.backward()
    grad = model.fc0.weight.grad
    assert grad is not None
    assert not torch.isnan(grad).any()
