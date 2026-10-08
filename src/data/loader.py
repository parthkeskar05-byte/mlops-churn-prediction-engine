"""Data loading utilities."""

import pandas as pd
from pathlib import Path
from typing import Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load telco churn dataset from CSV.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    required_cols = ['Churn', 'tenure', 'MonthlyCharges']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"✓ Loaded {len(df)} records with {len(df.columns)} features")
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found")

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    logger.info(f"Split data: X shape {X.shape}, y shape {y.shape}")
    return X, y
