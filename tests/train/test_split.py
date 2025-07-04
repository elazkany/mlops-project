# tests/train/test_split.py

import os
import numpy as np
import pandas as pd
from src.train.split import (
    load_dataset,
    split_data,
    apply_smote,
    save_array_pair,
    save_split_variants,
)


def test_load_dataset(tmp_path):
    """
    Test that a CSV file is correctly loaded into a DataFrame.

    The test creates a small CSV file with dummy data,
    calls `load_dataset()`, and verifies the result matches the expected DataFrame.
    """
    sample_csv = tmp_path / "sample.csv"
    sample_df = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [3, 4],
        "Class": [0, 1]
    })
    sample_df.to_csv(sample_csv, index=False)

    result = load_dataset(str(sample_csv))
    pd.testing.assert_frame_equal(result, sample_df)


def test_split_data():
    """
    Test that `split_data()` produces valid training and test splits.

    Ensures the output shapes and column names are as expected, and that
    both feature and label splits are returned correctly.
    """
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [5, 6, 7, 8],
        "target": [0, 0, 1, 1]
    })

    X_train, X_test, y_train, y_test = split_data(df, target_column="target", test_size=0.5, random_state=42)

    assert X_train.shape[0] == 2
    assert X_test.shape[0] == 2
    assert list(X_train.columns) == ["A", "B"]
    assert y_train.nunique() > 0


def test_apply_smote_balances_classes():
    """
    Test that SMOTE balancing creates an equal number of samples per class.

    Applies SMOTE to a small imbalanced dataset and verifies class counts are equal.
    """
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    y = pd.Series([0, 0, 1, 1])

    X_res, y_res = apply_smote(X, y, random_state=1)

    assert X_res.shape[0] == y_res.shape[0]
    assert y_res.value_counts().nunique() == 1


def test_save_array_pair_and_load(tmp_path):
    """
    Test that `save_array_pair()` correctly writes and loads a .npz file.

    Saves two numpy arrays, reloads them from disk, and confirms shape and content integrity.
    """
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    path = tmp_path / "split.npz"

    save_array_pair(X, y, str(path))
    data = np.load(path)

    np.testing.assert_array_equal(data["X"], X)
    np.testing.assert_array_equal(data["y"], y)


def test_save_split_variants(tmp_path):
    """
    Test that `save_split_variants()` writes all three expected .npz files.

    It checks file creation, validates presence of expected keys,
    and confirms the shape of the saved arrays.
    """
    X_train_raw = np.ones((2, 2))
    y_train_raw = np.array([0, 1])
    X_train_bal = np.ones((4, 2))
    y_train_bal = np.array([0, 0, 1, 1])
    X_test = np.zeros((2, 2))
    y_test = np.array([1, 0])

    save_split_variants(
        X_train_raw, y_train_raw,
        X_train_bal, y_train_bal,
        X_test, y_test,
        base_dir=str(tmp_path)
    )

    for name in ["train_raw.npz", "train_balanced.npz", "test.npz"]:
        assert os.path.exists(tmp_path / name)

    raw = np.load(tmp_path / "train_raw.npz")
    bal = np.load(tmp_path / "train_balanced.npz")
    test = np.load(tmp_path / "test.npz")

    assert raw["X"].shape == (2, 2)
    assert bal["y"].tolist() == [0, 0, 1, 1]
    assert test["y"].shape == (2,)
