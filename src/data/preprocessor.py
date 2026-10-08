"""Feature engineering and preprocessing."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChurnPreprocessor:
    """Preprocess telco churn data for modeling."""

    def __init__(self):
        """Initialize preprocessor with encoders and scaler."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> 'ChurnPreprocessor':
        """
        Fit encoders and scaler on training data.

        Args:
            X: Training features DataFrame

        Returns:
            Self for method chaining
        """
        X = X.copy()

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Fitting preprocessor on {len(categorical_cols)} categorical and {len(numerical_cols)} numerical features")

        # Fit label encoders for categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le

        # Transform categorical features
        X_encoded = X.copy()
        for col in categorical_cols:
            X_encoded[col] = self.label_encoders[col].transform(X[col].astype(str))

        # Fit scaler on numerical features
        if numerical_cols:
            self.scaler.fit(X_encoded[numerical_cols])

        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        logger.info("✓ Preprocessor fitted")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted encoders and scaler.

        Args:
            X: Features DataFrame to transform

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If preprocessor not fitted
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        X = X.copy()

        # Ensure all expected columns present
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Reorder columns to match training
        X = X[self.feature_names]

        # Transform categorical features
        categorical_cols = [col for col in X.columns if col in self.label_encoders]
        for col in categorical_cols:
            X[col] = self.label_encoders[col].transform(X[col].astype(str))

        # Transform numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Features DataFrame

        Returns:
            Transformed DataFrame
        """
        return self.fit(X).transform(X)
