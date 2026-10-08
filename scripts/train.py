"""Training script entrypoint."""

import sys
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.loader import load_data, split_features_target
from src.data.preprocessor import ChurnPreprocessor
from src.model.trainer import ChurnModelTrainer
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting MLOps Churn Prediction Training Pipeline")
    logger.info("=" * 60)

    # Load configuration
    config_path = Path(__file__).parent / ".." / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Generate sample data if it doesn't exist
    data_path = Path(__file__).parent / ".." / config['paths']['data_path']
    if not data_path.exists():
        logger.info("Generating sample data...")
        sys.path.insert(0, str(data_path.parent))
        from generate_sample import generate_telco_churn_data
        df = generate_telco_churn_data(config['data']['n_samples'])
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"✓ Generated {len(df)} samples")

    # Load data
    df = load_data(str(data_path))
    X, y = split_features_target(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )
    logger.info(f"✓ Split: Train {len(X_train)}, Test {len(X_test)}")

    # Preprocess
    preprocessor = ChurnPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    logger.info("✓ Preprocessing complete")

    # Train model
    trainer = ChurnModelTrainer(config)
    model = trainer.train(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        experiment_name=config['mlflow']['experiment_name']
    )

    # Save model
    model_path = Path(__file__).parent / ".." / config['paths']['model_path']
    trainer.save_model(str(model_path))

    # Save preprocessor
    preprocessor_path = model_path.parent / "preprocessor.pkl"
    import joblib
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"✓ Preprocessor saved to {preprocessor_path}")

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Accuracy: {trainer.metrics['accuracy']:.4f}")
    logger.info(f"MLflow tracking URI: {config['mlflow']['tracking_uri']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
