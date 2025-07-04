# tests/utils/test_io_save.py

import json
import numpy as np
import pandas as pd
from src.utils.io_save import (
    save_dataframe,
    save_npz,
    save_metrics,
    save_predictions,
    save_roc_curve
)


def test_save_dataframe(tmp_path):
    df = pd.DataFrame({
        "col1": [1, 2],
        "col2": ["a", "b"]
    })

    path = tmp_path / "data" / "output.csv"
    save_dataframe(df, str(path))

    assert path.exists()
    df_loaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(df, df_loaded)


def test_save_npz(tmp_path):
    X = np.random.rand(5, 3)
    y = np.array([1, 0, 1, 1, 0])
    path = tmp_path / "arrays" / "data.npz"

    save_npz(X, y, str(path))

    assert path.exists()
    data = np.load(path)
    assert np.allclose(X, data["X"])
    assert np.array_equal(y, data["y"])


def test_save_metrics(tmp_path, monkeypatch):
    # Override cwd to tmp_path so metrics.json is written there
    monkeypatch.chdir(tmp_path)

    metrics = {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.91,
        "f1_score": 0.915
    }

    save_metrics(metrics)

    path = tmp_path / "metrics.json"
    assert path.exists()

    with open(path) as f:
        loaded = json.load(f)
    assert loaded == metrics


def test_save_predictions(tmp_path):
    y_test = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 0])
    path = tmp_path / "plots" / "predictions.csv"

    save_predictions(y_test, y_pred, str(path))

    assert path.exists()

    df = pd.read_csv(path)
    assert list(df.columns) == ["true_label", "predicted_label"]
    assert df.shape == (3, 2)
    assert df["true_label"].tolist() == [1, 0, 1]
    assert df["predicted_label"].tolist() == [1, 0, 0]


def test_save_roc_curve(tmp_path):
    y_test = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([
        [0.8, 0.2],
        [0.6, 0.4],
        [0.3, 0.7],
        [0.1, 0.9],
    ])
    path = tmp_path / "plots" / "roc_curve.csv"

    save_roc_curve(y_test, y_pred_proba, str(path))

    assert path.exists()
    df = pd.read_csv(path)
    assert list(df.columns) == ["fpr", "tpr"]
    assert len(df) > 0
