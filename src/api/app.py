"""FastAPI application for churn prediction."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import time
from pathlib import Path
import yaml

from src.api.schemas import CustomerFeatures, ChurnPrediction, HealthResponse, MetricsResponse
from src.data.preprocessor import ChurnPreprocessor
from src.model.predictor import ChurnPredictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="High-concurrency FastAPI endpoint for customer churn prediction",
    version="1.0.0"
)

# Load configuration
config_path = Path(__file__).parents[2] / "config" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
predictor: ChurnPredictor = None
preprocessor: ChurnPreprocessor = None
metrics = {
    "total_predictions": 0,
    "total_latency_ms": 0.0
}


@app.on_event("startup")
async def load_model():
    """Load model and preprocessor on startup."""
    global predictor, preprocessor

    try:
        model_path = Path(__file__).parents[2] / config['paths']['model_path']

        if model_path.exists():
            predictor = ChurnPredictor(str(model_path))
            preprocessor = ChurnPreprocessor()

            # Fit preprocessor on sample data for inference-time preprocessing
            data_path = Path(__file__).parents[2] / config['paths']['data_path']
            if data_path.exists():
                df = pd.read_csv(data_path)
                X = df.drop('Churn', axis=1)
                preprocessor.fit(X)

            logger.info("✓ Model and preprocessor loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}. Run training first.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None
    )


@app.post("/predict", response_model=ChurnPrediction, tags=["Prediction"])
async def predict_churn(features: CustomerFeatures):
    """
    Predict customer churn probability.

    Args:
        features: Customer feature data

    Returns:
        Churn prediction with probability and risk level

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if predictor is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    start_time = time.time()

    try:
        # Convert to DataFrame
        features_dict = features.dict()
        df = pd.DataFrame([features_dict])

        # Preprocess
        df_processed = preprocessor.transform(df)

        # Predict
        result = predictor.predict_single(df_processed.to_dict('records')[0])

        # Update metrics
        latency_ms = (time.time() - start_time) * 1000
        metrics["total_predictions"] += 1
        metrics["total_latency_ms"] += latency_ms

        logger.info(f"Prediction completed in {latency_ms:.2f}ms")

        return ChurnPrediction(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """
    Get API performance metrics.

    Returns:
        Metrics including total predictions and average latency
    """
    avg_latency = (
        metrics["total_latency_ms"] / metrics["total_predictions"]
        if metrics["total_predictions"] > 0
        else 0.0
    )

    return MetricsResponse(
        total_predictions=metrics["total_predictions"],
        avg_latency_ms=round(avg_latency, 2)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=True
    )
