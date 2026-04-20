#!/usr/bin/env python3
"""Generate a small demo checkpoint for the FastAPI inference endpoint.

Trains a 3-member PINNEnsemble for 100 epochs on synthetic pump data and
saves it to models/ensemble.pt. The API server loads this automatically.

This is intentionally lightweight (not production quality). For full training
use scripts/train_pump_surrogate.py with the default config (600 epochs, 10 members).

Usage:
    uv run python scripts/generate_demo_checkpoint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics_models.data_generator import generate_pump_field_data
from src.physics_models.pump import PumpParameters, PumpPhysics
from src.surrogates.ensemble import PINNEnsemble
from src.surrogates.pinn import PINNConfig

CHECKPOINT = Path("models/ensemble.pt")


def main() -> None:
    print("Generating synthetic pump data...")
    df = generate_pump_field_data(n_samples=300, seed=42)

    params = PumpParameters(
        design_flow=0.05,
        design_head=30.0,
        design_speed=1450.0,
        design_efficiency=0.75,
    )
    physics = PumpPhysics(params)

    config = PINNConfig(
        hidden_dims=[32, 32],
        lambda_physics=0.01,
        lr=1e-3,
        mse_warmup_epochs=20,
    )

    ensemble = PINNEnsemble(config=config, n_members=3, physics=physics)

    import numpy as np

    X = df[["flow_rate", "speed", "operating_hours"]].to_numpy(dtype=np.float32)
    y = df["head"].to_numpy(dtype=np.float32).reshape(-1, 1)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    print("Training 3-member ensemble (100 epochs)...")
    ensemble.fit(X_t, y_t, max_epochs=100, batch_size=64, accelerator="cpu")

    CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save(ensemble, CHECKPOINT)
    print(f"Saved → {CHECKPOINT}")

    pred = ensemble.predict(X_t[:5])
    print(f"Sample predictions (head m): {pred.mean[:, 0].tolist()}")
    print(f"Total uncertainty:           {pred.total_std[:, 0].tolist()}")


if __name__ == "__main__":
    main()
