# src/utils/io_save.py

import os
import json

from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to CSV format.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_npz(X: np.ndarray, y: np.ndarray, path: str) -> None:
    """
    Save X and y arrays to a compressed NPZ file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X, y=y)


def save_metrics(metrics: Dict[str, Union[float, int]]):
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


def save_predictions(y_test: np.ndarray, y_pred: np.ndarray, path: str = "plots/predictions.csv") -> None:
    """
    Save true and predicted labels to a CSV file for evaluation (e.g., confusion matrix).

    Parameters:
        y_test (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        path (str): File path to save the CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame(
        np.column_stack([y_test, y_pred]),
        columns=["true_label", "predicted_label"]
    ).astype(int)

    df.to_csv(path, index=False)


def save_roc_curve(y_test: np.ndarray, y_pred_proba: np.ndarray, path: str = "plots/roc_curve.csv") -> None:
    """
    Calculate and save ROC curve data (FPR and TPR) to CSV.

    Parameters:
        y_test (np.ndarray): True binary labels.
        y_pred_proba (np.ndarray): Prediction probabilities (2nd column = positive class).
        path (str): File path to save the CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    df = pd.DataFrame(
        np.column_stack([fpr, tpr]),
        columns=["fpr", "tpr"]
    )

    df.to_csv(path, index=False)
