"""
EduPredict Cloud — Tests: Health Endpoint
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_shape():
    resp = client.get("/health")
    data = resp.json()
    assert "status" in data
    assert "model_loaded" in data


def test_health_status_is_healthy():
    resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "healthy"


def test_health_model_loaded_is_bool():
    resp = client.get("/health")
    data = resp.json()
    assert isinstance(data["model_loaded"], bool)


def test_health_content_type():
    resp = client.get("/health")
    assert "application/json" in resp.headers.get("content-type", "")
