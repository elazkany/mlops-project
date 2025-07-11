# src/train/run_train.py

from pathlib import Path
import mlflow
from src.utils.io_load import load_npz, load_params
from src.train.train_and_evaluate import create_and_train_model


def main():
    # Load experiment name from params.yaml
    experiment_name = load_params("experiment_tracking")["experiment_name"]

    # Load data
    X_train, y_train = load_npz("data/train/train_original.npz")
    X_train_resampled, y_train_resampled = load_npz("data/train/train_resampled.npz")
    X_valid, y_valid = load_npz("data/validation/valid.npz")

    mlruns_path = Path("mlruns")  # or = Path("mlruns").resolve() for absolute path
    mlflow.set_tracking_uri(f"file:./{mlruns_path}")

    # Model configs to train
    configs = [
        (
            "logistic_regression",
            "lr_without_smote",
            X_train,
            y_train
        ),
        (
            "logistic_regression",
            "lr_with_smote",
            X_train_resampled,
            y_train_resampled
        ),
        (
            "random_forest",
            "rf_without_smote",
            X_train,
            y_train
        ),
        (
            "random_forest",
            "rf_with_smote",
            X_train_resampled,
            y_train_resampled
        ),
    ]

    # Train and log each model
    for model_type, model_name, X, y in configs:
        create_and_train_model(
            model_type=model_type,
            model_name=model_name,
            X_train=X,
            y_train=y,
            X_valid=X_valid,
            y_valid=y_valid,
            experiment_name=experiment_name
        )


if __name__ == "__main__":
    main()
