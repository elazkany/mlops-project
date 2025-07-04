"""
Test suite for split.py functions, including SMOTE resampling.
"""

import numpy as np
import pandas as pd
from src.train.split import split_data, apply_smote


def test_split_data_output_shapes():
    """
    Test that split_data correctly splits data and maintains expected shapes.
    """
    df = pd.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "target": [0]*50 + [1]*50
    })

    X_train, X_test, y_train, y_test = split_data(
        df=df,
        target_column="target",
        test_size=0.2,
        random_state=42
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    assert "target" not in X_train.columns


def test_apply_smote_balances_classes():
    """
    Test that apply_smote successfully balances an imbalanced class distribution.
    """
    # Generate imbalanced dummy data
    X = pd.DataFrame({
        "feature1": np.random.rand(60),
        "feature2": np.random.rand(60)
    })
    y = pd.Series([0]*50 + [1]*10)  # Minority class has 10 samples, SMOTE-safe

    # Apply SMOTE
    X_resampled, y_resampled = apply_smote(X, y, random_state=42)

    # Confirm new class balance
    counts = y_resampled.value_counts()
    assert counts[0] == counts[1]
    assert len(X_resampled) == len(y_resampled)
