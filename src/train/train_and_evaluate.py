# src/train/train_and_evaluate.py

import json
import joblib
import os
from src.utils.io_load import load_params, load_npz
from src.utils.io_save import save_metrics, save_predictions, save_roc_curve
from src.models.model import train_model, evaluate_model
from src.utils.plot_utils import plot_confusion_matrix


def main():
    # Load model params
    model_params = load_params("model")

    # Load data
    X_train, y_train = load_npz("data/split/train_balanced.npz")
    X_test, y_test = load_npz("data/split/test.npz")

    # Train model
    model = train_model(X_train, y_train, model_params)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)

    # Evaluate
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

    print(f"✅ Model trained and saved to {model_path}")
    print(f"📈 Metrics:\n{json.dumps(metrics, indent=4)}")

    save_metrics(metrics)
    save_predictions(y_test, y_pred)
    save_roc_curve(y_test, y_pred_proba)
    plot_confusion_matrix(model, X_test, y_test)


if __name__ == "__main__":
    main()
