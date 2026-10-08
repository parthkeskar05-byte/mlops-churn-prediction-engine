"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "healthy"


def test_metrics_endpoint():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "avg_latency_ms" in data


def test_predict_endpoint_schema():
    """Test prediction endpoint with valid schema."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20
    }

    response = client.post("/predict", json=payload)

    # Will return 503 if model not loaded, which is expected in test environment
    assert response.status_code in [200, 503]


def test_predict_endpoint_invalid_data():
    """Test prediction endpoint with invalid data."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0
        # Missing required fields
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
