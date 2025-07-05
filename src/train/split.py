# src/train/split.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.utils.io_load import load_params, load_dataset
from src.utils.io_save import save_npz


def split_data(df: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    """
    Split dataset into train/test sets.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Column name of the label.
        test_size (float): Proportion of test split.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def apply_smote(X, y, random_state):
    """
    Apply SMOTE to balance training data.

    Parameters:
        X (pd.DataFrame): Training features.
        y (pd.Series): Training labels.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple: Oversampled X and y.
    """
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X, y)


def save_split_variants(
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    X_train_balanced: np.ndarray,
    y_train_balanced: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_dir: str
) -> None:
    """
    Save raw, balanced, and test splits to their respective .npz files.
    """
    save_npz(X_train_raw, y_train_raw, os.path.join(base_dir, "train_raw.npz"))
    save_npz(X_train_balanced, y_train_balanced, os.path.join(base_dir, "train_balanced.npz"))
    save_npz(X_test, y_test, os.path.join(base_dir, "test.npz"))


def main():
    params = load_params("split")

    df = load_dataset(params["processed_data_path"])
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=params["target_column"],
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, random_state=params["random_state"])

    save_split_variants(
        X_train_raw=X_train,
        y_train_raw=y_train,
        X_train_balanced=X_train_resampled,
        y_train_balanced=y_train_resampled,
        X_test=X_test,
        y_test=y_test,
        base_dir=params["split_data_path"]
    )


if __name__ == "__main__":
    main()
