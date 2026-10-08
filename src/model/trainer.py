"""XGBoost model training with MLflow tracking."""

import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Tuple, Any
import joblib
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChurnModelTrainer:
    """Train XGBoost model with MLflow experiment tracking."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.

        Args:
            config: Configuration dictionary with model params
        """
        self.config = config
        self.model = None
        self.metrics: Dict[str, float] = {}

    def train(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        experiment_name: str = "churn-prediction"
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model with MLflow tracking.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            experiment_name: MLflow experiment name

        Returns:
            Trained XGBoost classifier
        """
        mlflow.set_experiment(experiment_name)

        logger.info("Starting model training")

        with mlflow.start_run():
            # Model parameters
            params = self.config['model']
            mlflow.log_params(params)

            # Train model
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }

            # Log metrics
            mlflow.log_metrics(self.metrics)

            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")

            # Log model
            mlflow.xgboost.log_model(self.model, "model")

            logger.info(f"✓ Training complete - Accuracy: {self.metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {self.metrics['precision']:.4f}, Recall: {self.metrics['recall']:.4f}")
            logger.info(f"  F1: {self.metrics['f1_score']:.4f}, ROC-AUC: {self.metrics['roc_auc']:.4f}")

        return self.model

    def save_model(self, path: str) -> None:
        """
        Save trained model to disk.

        Args:
            path: Output file path

        Raises:
            ValueError: If model not trained yet
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, output_path)
        logger.info(f"✓ Model saved to {output_path}")

    def load_model(self, path: str) -> xgb.XGBClassifier:
        """
        Load trained model from disk.

        Args:
            path: Model file path

        Returns:
            Loaded XGBoost classifier
        """
        self.model = joblib.load(path)
        logger.info(f"✓ Model loaded from {path}")
        return self.model
