import pandas as pd
import os
import yaml
from src.data.preprocess import load_params, read_dataset, scale_columns, save_processed_data


def test_load_params(tmp_path):
    """
    Test if YAML parameters are correctly loaded.
    """
    dummy_yaml = tmp_path / "params.yaml"
    content = {
        "preprocess": {
            "raw_data_path": "data/raw/sample.csv",
            "processed_data_path": "data/processed/sample_out.csv",
            "scale_columns": ["Amount", "Time"],
        }
    }
    with open(dummy_yaml, "w") as f:
        yaml.dump(content, f)

    params = load_params(str(dummy_yaml))
    assert params["raw_data_path"] == "data/raw/sample.csv"
    assert "Amount" in params["scale_columns"]


def test_read_dataset(tmp_path):
    """
    Test if a CSV file is read correctly into a DataFrame.
    """
    test_data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    file_path = tmp_path / "test.csv"
    test_data.to_csv(file_path, index=False)

    df = read_dataset(str(file_path))
    pd.testing.assert_frame_equal(df, test_data)


def test_scale_columns():
    """
    Test if specified columns are scaled using RobustScaler.
    """
    df = pd.DataFrame({"Amount": [1, 100, 200], "Time": [10, 1000, 3000]})
    df_scaled = scale_columns(df, ["Amount", "Time"])
    assert abs(df_scaled["Amount"].mean()) < 1
    assert "Amount" in df_scaled.columns


def test_save_processed_data(tmp_path):
    """
    Test if a DataFrame is saved correctly to CSV format.
    """
    df = pd.DataFrame({"X": [1, 2, 3]})
    file_path = tmp_path / "out" / "result.csv"
    save_processed_data(df, str(file_path))

    assert os.path.exists(file_path)
    loaded_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(df, loaded_df)
