"""Tests for model training and prediction."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from src.model.trainer import ChurnModelTrainer
from src.model.predictor import ChurnPredictor


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    n = 100

    X_train = pd.DataFrame({
        'feature1': np.random.rand(n),
        'feature2': np.random.randint(0, 2, n),
        'feature3': np.random.rand(n) * 100
    })
    X_test = pd.DataFrame({
        'feature1': np.random.rand(20),
        'feature2': np.random.randint(0, 2, 20),
        'feature3': np.random.rand(20) * 100
    })

    y_train = pd.Series(np.random.randint(0, 2, n))
    y_test = pd.Series(np.random.randint(0, 2, 20))

    return X_train, X_test, y_train, y_test


def test_trainer_initialization():
    """Test trainer initialization."""
    config = {
        'model': {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }

    trainer = ChurnModelTrainer(config)
    assert trainer.model is None
    assert trainer.config == config


def test_model_training(sample_training_data):
    """Test model training."""
    X_train, X_test, y_train, y_test = sample_training_data

    config = {
        'model': {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }

    trainer = ChurnModelTrainer(config)
    model = trainer.train(X_train, X_test, y_train, y_test)

    assert model is not None
    assert trainer.metrics is not None
    assert 'accuracy' in trainer.metrics


def test_model_save_load(sample_training_data):
    """Test model saving and loading."""
    X_train, X_test, y_train, y_test = sample_training_data

    config = {
        'model': {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }

    trainer = ChurnModelTrainer(config)
    trainer.train(X_train, X_test, y_train, y_test)

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        model_path = f.name

    try:
        trainer.save_model(model_path)
        assert Path(model_path).exists()

        new_trainer = ChurnModelTrainer(config)
        loaded_model = new_trainer.load_model(model_path)
        assert loaded_model is not None
    finally:
        Path(model_path).unlink()


def test_predictor_initialization():
    """Test predictor initialization with invalid path."""
    with pytest.raises(FileNotFoundError):
        ChurnPredictor("nonexistent_model.pkl")
