"""Tests for data preprocessing."""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import ChurnPreprocessor


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'tenure': [12, 24, 6],
        'MonthlyCharges': [50.0, 75.0, 25.0],
        'Contract': ['Month-to-month', 'One year', 'Two year']
    })


def test_preprocessor_fit(sample_data):
    """Test preprocessor fitting."""
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(sample_data)

    assert preprocessor.is_fitted
    assert len(preprocessor.label_encoders) > 0
    assert preprocessor.scaler is not None


def test_preprocessor_transform(sample_data):
    """Test preprocessor transformation."""
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(sample_data)

    transformed = preprocessor.transform(sample_data)

    assert transformed.shape == sample_data.shape
    assert isinstance(transformed, pd.DataFrame)


def test_preprocessor_fit_transform(sample_data):
    """Test preprocessor fit_transform."""
    preprocessor = ChurnPreprocessor()
    transformed = preprocessor.fit_transform(sample_data)

    assert transformed.shape == sample_data.shape
    assert preprocessor.is_fitted


def test_preprocessor_not_fitted():
    """Test error when transforming before fitting."""
    preprocessor = ChurnPreprocessor()
    sample_data = pd.DataFrame({'feature': [1, 2, 3]})

    with pytest.raises(ValueError, match="must be fitted"):
        preprocessor.transform(sample_data)
