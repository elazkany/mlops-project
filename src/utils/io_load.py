# src/utils/io_load.py

import yaml
import numpy as np
import pandas as pd
from typing import Tuple


def load_params(section: str, path: str = "params.yaml") -> dict:
    """
    Load a specific section from the params.yaml config file.

    Parameters:
        section (str): Top-level section name in the YAML.
        path (str): Path to the YAML file.

    Returns:
        dict: Dictionary containing the requested section’s parameters.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config.get(section, {})


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset into a Pandas DataFrame.
    """
    return pd.read_csv(path)


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load X and y from a .npz file.
    """
    data = np.load(path)
    return data["X"], data["y"]
