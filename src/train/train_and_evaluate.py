# src/train/train_and_evaluate.py

import os
import mlflow
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
from src.utils.io_load import load_params
from src.models.model import train_model, evaluate_model
from sklearn.metrics import ConfusionMatrixDisplay


def create_and_train_model(
    model_type: str,             # e.g., "logistic_regression" or "random_forest"
    model_name: str,            # e.g., "lr-without-SMOTE" - no spaces
    X_train,
    y_train,
    X_valid,
    y_valid,
    experiment_name: str = "credit-card-fraud-detection"
) -> str:
    """
    Train, evaluate, and log a model using MLflow.
    """
    # === Load parameters from params.yaml ===
    model_config = load_params("models")
    model_params = model_config[model_type]["model_params"]

    # === Train the model ===
    model = train_model(
        X_train,
        y_train,
        model_type=model_type,
        model_params=model_params
    )

    # === Evaluate the model ===
    metrics, _, _ = evaluate_model(model, X_valid, y_valid)

    # === Inferred MLflow signature ===
    input_example = X_train[:1]
    signature = infer_signature(X_train, model.predict(X_train))

    if mlflow.active_run():
        mlflow.end_run()

    # === Set MLflow experiment ===
    mlflow.set_experiment(experiment_name)  # where to log the next run

    print(f"{model_name} trained and evaluated")

    with mlflow.start_run(run_name=model_name):

        # Log metadata
        mlflow.set_tags({
            "model_type": model_type,
            "run_name": model_name,
        })

        # Log hyperparameters and metrics
        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)

        # Log the model
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            signature=signature
        )

        # Plot and log confusion matrix
        ConfusionMatrixDisplay.from_estimator(model, X_valid, y_valid)
        plt.title(f"{model_name}\nConfusion Matrix")
        plot_path = "confusion_matrix.png"  # The name should be the same to compare commom artifact
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        os.remove(plot_path)  # Remove it from root it's already in mlruns

    print(f"{model_name} model and artifacts logged by MLflow")
