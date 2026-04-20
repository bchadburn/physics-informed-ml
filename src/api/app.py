"""FastAPI inference endpoint for PINN surrogate predictions.

Predicts pump head with uncertainty decomposition given operating conditions.
Loads a trained PINNEnsemble from models/ensemble.pt if available;
falls back to physics-model predictions when no checkpoint is present.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.physics_models.pump import PumpParameters, PumpPhysics
from src.surrogates.ensemble import PINNEnsemble

CHECKPOINT_PATH = Path("models/ensemble.pt")

# Module-level state — loaded once on startup
_ensemble: PINNEnsemble | None = None
_physics: PumpPhysics | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _ensemble, _physics
    params = PumpParameters(
        design_flow=0.05,
        design_head=30.0,
        design_speed=1450.0,
        design_efficiency=0.75,
    )
    _physics = PumpPhysics(params)
    if CHECKPOINT_PATH.exists():
        _ensemble = torch.load(CHECKPOINT_PATH, weights_only=False)
    yield


app = FastAPI(title="PINN Surrogate API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    flow_rate: float = Field(..., gt=0, description="Flow rate (m³/s)")
    speed: float = Field(..., gt=0, description="Pump speed (rpm)")
    operating_hours: float = Field(..., ge=0, description="Cumulative operating hours")


class PredictResponse(BaseModel):
    head_mean: float = Field(..., description="Predicted head (m)")
    epistemic_std: float | None = Field(None, description="Model uncertainty (m); None when using physics fallback")
    aleatoric_std: float | None = Field(None, description="Data noise (m); None when using physics fallback")
    total_std: float | None = Field(None, description="Total uncertainty (m); None when using physics fallback")
    source: str = Field(..., description="'ensemble' or 'physics_fallback'")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "ensemble" if _ensemble is not None else "physics_fallback"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _ensemble is not None:
        X = torch.tensor([[req.flow_rate, req.speed, req.operating_hours]], dtype=torch.float32)
        pred = _ensemble.predict(X)
        return PredictResponse(
            head_mean=float(pred.mean[0, 0]),
            epistemic_std=float(pred.epistemic_std[0, 0]),
            aleatoric_std=float(pred.aleatoric_std[0, 0]),
            total_std=float(pred.total_std[0, 0]),
            source="ensemble",
        )
    # Physics fallback — vendor curve, no uncertainty
    head = _physics.head(req.flow_rate, req.speed)  # type: ignore[union-attr]
    return PredictResponse(
        head_mean=head,
        epistemic_std=None,
        aleatoric_std=None,
        total_std=None,
        source="physics_fallback",
    )
