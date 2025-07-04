# tests/data/test_preprocess.py

import pandas as pd
from src.data.preprocess import scale_columns


def test_scale_columns():
    """
    Test if specified columns are scaled using RobustScaler.
    """
    df = pd.DataFrame({
        "Amount": [1, 100, 200],
        "Time": [10, 1000, 3000]
        })
    df_scaled = scale_columns(df, ["Amount", "Time"])
    assert abs(df_scaled["Amount"].mean()) < 1
    # assert "Amount" in df_scaled.columns
    assert abs(df_scaled["Time"].median()) < 1

    # Confirm that scaling altered the original values
    assert not df_scaled.equals(df)
