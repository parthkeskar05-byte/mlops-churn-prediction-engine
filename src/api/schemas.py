"""Pydantic request and response schemas."""

from pydantic import BaseModel, Field
from typing import Optional


class CustomerFeatures(BaseModel):
    """Customer feature input schema."""

    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen status (0/1)")
    Partner: str = Field(..., description="Has partner (Yes/No)")
    Dependents: str = Field(..., description="Has dependents (Yes/No)")
    tenure: int = Field(..., ge=0, description="Months with company")
    PhoneService: str = Field(..., description="Has phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Multiple lines status")
    InternetService: str = Field(..., description="Internet service type")
    OnlineSecurity: str = Field(..., description="Online security service")
    OnlineBackup: str = Field(..., description="Online backup service")
    DeviceProtection: str = Field(..., description="Device protection service")
    TechSupport: str = Field(..., description="Tech support service")
    StreamingTV: str = Field(..., description="Streaming TV service")
    StreamingMovies: str = Field(..., description="Streaming movies service")
    Contract: str = Field(..., description="Contract type")
    PaperlessBilling: str = Field(..., description="Paperless billing (Yes/No)")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class ChurnPrediction(BaseModel):
    """Churn prediction response schema."""

    churn_prediction: int = Field(..., description="Predicted churn (0=No, 1=Yes)")
    churn_probability: float = Field(..., ge=0, le=1, description="Churn probability")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Model availability")


class MetricsResponse(BaseModel):
    """Metrics response schema."""

    total_predictions: int = Field(..., description="Total predictions served")
    avg_latency_ms: float = Field(..., description="Average latency in milliseconds")
