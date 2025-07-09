# src/models/model.py

import json
import numpy as np
from typing import Dict, Any, Tuple, Union
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def initialize_model(model_type: str, model_params: Dict[str, Any]) -> ClassifierMixin:
    """
    Create a model instance based on type and parameters.
    """
    if model_type == "random_forest":
        return RandomForestClassifier(**model_params)
    elif model_type == "logistic_regression":
        return LogisticRegression(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    model_params: Dict[str, Any]
) -> ClassifierMixin:
    """
    Train specified model type.

    Returns:
        Trained model
    """
    model = initialize_model(model_type, model_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
    float_precision: int = 4
) -> Tuple[Dict[str, float], np.ndarray, Union[np.ndarray, None]]:
    """
    Evaluate model and return classification metrics.

    Returns:
        metrics dict, predicted labels, predicted probabilities (if available)
    """
    y_pred = model.predict(X_test)

    # Use predict_proba if available
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    metrics = json.loads(
        json.dumps(metrics),
        parse_float=lambda x: round(float(x), float_precision)
    )

    return metrics, y_pred, y_proba
