# tests/utils/test_io_load.py

import yaml
import numpy as np
import pandas as pd
from src.utils.io_load import load_params, load_dataset, load_npz


def test_load_params(tmp_path):
    """
    Test that load_params reads the correct section from a YAML file.
    """
    yaml_content = {
        "model": {"n_estimators": 100, "max_depth": 5},
        "train": {"test_size": 0.2}
    }

    yaml_path = tmp_path / "params.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    model_params = load_params("model", path=str(yaml_path))
    assert model_params == {"n_estimators": 100, "max_depth": 5}

    # Nonexistent section returns empty dict
    unknown = load_params("nonexistent", path=str(yaml_path))
    assert unknown == {}


def test_load_dataset(tmp_path):
    """
    Test that load_dataset reads a CSV file into a DataFrame.
    """
    csv_path = tmp_path / "data.csv"
    df_original = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [3, 4],
        "target": [0, 1]
    })
    df_original.to_csv(csv_path, index=False)

    df_loaded = load_dataset(str(csv_path))
    pd.testing.assert_frame_equal(df_original, df_loaded)


def test_load_npz(tmp_path):
    """
    Test that load_npz loads arrays correctly from a .npz file.
    """
    npz_path = tmp_path / "data.npz"
    X = np.random.rand(3, 2)
    y = np.array([0, 1, 0])
    np.savez(npz_path, X=X, y=y)

    X_loaded, y_loaded = load_npz(str(npz_path))
    assert np.allclose(X, X_loaded)
    assert np.array_equal(y, y_loaded)
