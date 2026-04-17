# tests/test_pinn.py
import torch
import pytest
from src.surrogates.pinn import PINN, PINNConfig


def _make_batch(n: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X, y) where X = [flow, speed, hours], y = head."""
    X = torch.rand(n, 3)
    X[:, 0] *= 0.1    # flow: 0–0.1 m³/s
    X[:, 1] = X[:, 1] * 1000 + 1000   # speed: 1000–2000 rpm
    X[:, 2] *= 8760   # hours: 0–8760
    y = torch.rand(n, 1) * 40 + 5     # head: 5–45 m
    return X, y


def test_pinn_forward_shape():
    """Forward pass returns (mean, logvar) both shape [N, 1]."""
    config = PINNConfig(hidden_dims=[32, 32], activation="tanh")
    model = PINN(config)
    X, _ = _make_batch(8)
    mean, logvar = model(X)
    assert mean.shape == (8, 1), f"Expected (8,1), got {mean.shape}"
    assert logvar.shape == (8, 1), f"Expected (8,1), got {logvar.shape}"


def test_pinn_loss_has_data_and_physics_components():
    """Loss dict must contain 'loss_data' and 'loss_physics' keys."""
    from src.physics_models.pump import PumpParameters, PumpPhysics
    config = PINNConfig(hidden_dims=[32, 32], activation="tanh",
                        lambda_data=1.0, lambda_physics=0.1)
    physics = PumpPhysics(PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    ))
    model = PINN(config, physics=physics)
    X, y = _make_batch(16)
    X.requires_grad_(True)
    losses = model.compute_losses(X, y)
    assert "loss_data" in losses
    assert "loss_physics" in losses
    assert "loss_total" in losses


def test_pinn_physics_loss_shape_independent_of_batch():
    """Physics loss should be a scalar regardless of batch size."""
    from src.physics_models.pump import PumpParameters, PumpPhysics
    config = PINNConfig(hidden_dims=[32, 32], activation="tanh",
                        lambda_data=1.0, lambda_physics=0.1)
    physics = PumpPhysics(PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    ))
    model = PINN(config, physics=physics)
    for n in [8, 32, 128]:
        X, y = _make_batch(n)
        X.requires_grad_(True)
        losses = model.compute_losses(X, y)
        assert losses["loss_physics"].ndim == 0, "Physics loss must be scalar"


def test_pinn_training_step_decreases_loss():
    """After 50 gradient steps, total loss should decrease."""
    from src.physics_models.pump import PumpParameters, PumpPhysics
    config = PINNConfig(hidden_dims=[32, 32], activation="tanh",
                        lambda_data=1.0, lambda_physics=0.1, lr=1e-3)
    physics = PumpPhysics(PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    ))
    model = PINN(config, physics=physics)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X, y = _make_batch(64)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        X_grad = X.clone().requires_grad_(True)
        loss_dict = model.compute_losses(X_grad, y)
        loss_dict["loss_total"].backward()
        optimizer.step()
        losses.append(loss_dict["loss_total"].item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
    )


from src.surrogates.ensemble import PINNEnsemble, EnsemblePrediction


def test_ensemble_predict_returns_uncertainty():
    """Ensemble prediction must return mean, epistemic_std, aleatoric_std."""
    from src.physics_models.pump import PumpParameters, PumpPhysics
    from src.surrogates.pinn import PINNConfig
    config = PINNConfig(hidden_dims=[16, 16], activation="tanh")
    physics = PumpPhysics(PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    ))
    ensemble = PINNEnsemble(config=config, n_members=3, physics=physics)
    X, _ = _make_batch(8)
    pred = ensemble.predict(X)
    assert isinstance(pred, EnsemblePrediction)
    assert pred.mean.shape == (8, 1)
    assert pred.epistemic_std.shape == (8, 1)
    assert pred.aleatoric_std.shape == (8, 1)
    assert pred.total_std.shape == (8, 1)


def test_ensemble_epistemic_uncertainty_increases_ood():
    """Epistemic uncertainty should be higher far outside training range."""
    from src.physics_models.pump import PumpParameters, PumpPhysics
    from src.surrogates.pinn import PINNConfig
    from src.physics_models.data_generator import generate_pump_field_data

    config = PINNConfig(hidden_dims=[32, 32], activation="tanh",
                        lambda_data=1.0, lambda_physics=0.0, lr=1e-3)
    physics = PumpPhysics(PumpParameters(
        design_flow=0.05, design_head=30.0,
        design_speed=1450.0, design_efficiency=0.75,
    ))
    ensemble = PINNEnsemble(config=config, n_members=3, physics=physics)

    # Minimal training data (in-distribution)
    df = generate_pump_field_data(n_samples=100, seed=0)
    X_tr = torch.tensor(
        df[["flow_rate", "speed", "operating_hours"]].values, dtype=torch.float32
    )
    y_tr = torch.tensor(df[["head"]].values, dtype=torch.float32)
    ensemble.fit(X_tr, y_tr, max_epochs=5, batch_size=32)

    # In-distribution: normal operating point
    X_in = torch.tensor([[0.05, 1450.0, 1000.0]])
    # OOD: extreme speed far outside training range
    X_ood = torch.tensor([[0.05, 5000.0, 1000.0]])

    pred_in = ensemble.predict(X_in)
    pred_ood = ensemble.predict(X_ood)

    assert pred_ood.epistemic_std.mean() > pred_in.epistemic_std.mean(), (
        "OOD epistemic uncertainty should exceed in-distribution uncertainty"
    )
