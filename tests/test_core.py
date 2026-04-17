import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from core.metrics import relative_l2_error, expected_calibration_error, inference_timer
from core.benchmark import BenchmarkResult, render_markdown_table


def test_relative_l2_perfect():
    pred = np.array([1.0, 2.0, 3.0])
    assert relative_l2_error(pred, pred) == 0.0


def test_relative_l2_known():
    pred = np.array([2.0, 2.0])
    true = np.array([1.0, 1.0])
    # ||[1,1]||_2 / ||[1,1]||_2 = 1.0
    assert abs(relative_l2_error(pred, true) - 1.0) < 1e-6


def test_ece_perfect_calibration():
    """Perfectly calibrated Gaussian should have ECE near 0."""
    rng = np.random.default_rng(0)
    n = 2000
    mean = rng.normal(0, 1, n)
    std = np.ones(n)
    true = mean + rng.normal(0, 1, n)  # noise matches predicted std
    ece = expected_calibration_error(mean, std, true, n_bins=10)
    assert ece < 0.05


def test_inference_timer_returns_positive():
    model = nn.Linear(4, 1)
    x = torch.randn(16, 4)
    t = inference_timer(model, x, n_runs=10)
    assert t > 0.0


def test_render_markdown_table():
    results = [
        BenchmarkResult(name="FNO", relative_l2=0.023, inference_time_ms=1.2, n_train_samples=1000),
        BenchmarkResult(name="UNet", relative_l2=0.041, inference_time_ms=0.9, n_train_samples=1000),
    ]
    table = render_markdown_table(results)
    assert "FNO" in table
    assert "UNet" in table
    assert "|" in table
