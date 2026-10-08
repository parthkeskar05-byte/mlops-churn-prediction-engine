"""Model inference logic."""

import pandas as pd
import numpy as np
from typing import Dict, List
import joblib
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChurnPredictor:
    """Handle model inference for churn prediction."""

    def __init__(self, model_path: str):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to saved model file

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        logger.info(f"✓ Predictor initialized with model from {model_path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn labels.

        Args:
            X: Preprocessed features DataFrame

        Returns:
            Array of predicted labels (0 or 1)
        """
        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probabilities.

        Args:
            X: Preprocessed features DataFrame

        Returns:
            Array of predicted probabilities for positive class
        """
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities

    def predict_single(self, features: Dict[str, any]) -> Dict[str, any]:
        """
        Predict churn for a single customer.

        Args:
            features: Dictionary of customer features

        Returns:
            Dictionary with prediction and probability
        """
        df = pd.DataFrame([features])

        prediction = self.predict(df)[0]
        probability = self.predict_proba(df)[0]

        result = {
            'churn_prediction': int(prediction),
            'churn_probability': float(probability),
            'risk_level': self._get_risk_level(probability)
        }

        return result

    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """
        Categorize churn risk based on probability.

        Args:
            probability: Churn probability

        Returns:
            Risk level string
        """
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Medium"
        else:
            return "High"
