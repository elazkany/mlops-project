# tests/test_models.py

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.models.model import train_model, evaluate_model


@pytest.fixture
def sample_data():
    """
    Returns a simple binary classification dataset for testing.
    """
    X_train = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    y_train = np.array([1, 0, 1, 0, 1, 0])
    return X_train, y_train


@pytest.mark.parametrize("model_type, model_class", [
    ("random_forest", RandomForestClassifier),
    ("logistic_regression", LogisticRegression)
])
def test_train_model(sample_data, model_type, model_class):
    """
    Test that the model trains and returns the correct model type instance.
    """
    X_train, y_train = sample_data
    model_params = {"n_estimators": 10, "random_state": 42} if model_type == "random_forest" else {"max_iter": 200}
    model = train_model(X_train, y_train, model_type, model_params)

    assert isinstance(model, model_class)
    assert hasattr(model, "predict")


@pytest.mark.parametrize("model_type", ["random_forest", "logistic_regression"])
def test_evaluate_model(sample_data, model_type):
    """
    Test that evaluation metrics are computed and returned correctly.
    """
    X, y = sample_data
    model_params = {"n_estimators": 10, "random_state": 42} if model_type == "random_forest" else {"max_iter": 200}
    model = train_model(X, y, model_type, model_params)

    metrics, y_pred, y_proba = evaluate_model(model, X, y)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1_score"}
    for metric in metrics.values():
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0

    assert isinstance(y_pred, np.ndarray)
    if y_proba is not None:
        assert isinstance(y_proba, np.ndarray)
        assert y_proba.shape[1] == 2  # For binary classification
