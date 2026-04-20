"""Tests for the FastAPI inference endpoint."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_physics_fallback_returns_positive_head(client):
    """With no checkpoint, the physics fallback should return a positive head."""
    resp = client.post("/predict", json={
        "flow_rate": 0.03,
        "speed": 1450.0,
        "operating_hours": 100.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["head_mean"] > 0
    assert body["source"] == "physics_fallback"
    assert body["epistemic_std"] is None


def test_predict_rejects_negative_flow(client):
    """flow_rate must be > 0."""
    resp = client.post("/predict", json={
        "flow_rate": -0.01,
        "speed": 1450.0,
        "operating_hours": 0.0,
    })
    assert resp.status_code == 422
