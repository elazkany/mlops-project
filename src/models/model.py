# src/models/model.py

import json
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, model_params: Dict[str, Any]
) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier using provided training data and parameters.

    Parameters:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        model_params (dict): Hyperparameters for the classifier.

    Returns:
        RandomForestClassifier: A trained model instance.
    """
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    float_precision: int = 4
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a trained classifier on test data.

    Parameters:
        model (RandomForestClassifier): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        float_precision (int): Decimal precision for metrics.

    Returns:
        dict: Accuracy, precision, recall, and F1 score.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    metrics = json.loads(
        json.dumps(metrics), parse_float=lambda x: round(float(x), float_precision)
    )

    return metrics, y_pred, y_proba
