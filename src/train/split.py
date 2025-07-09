# src/train/split.py

import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.utils.io_load import load_params, load_dataset
from src.utils.io_save import save_npz


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


def main():
    params = load_params("split")

    df = load_dataset(params["processed_data_path"])

    # First split: train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=params["target_column"]),
        df[params["target_column"]],
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    # Second split: train and validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    X_train_resampled, y_train_resampled = apply_smote(
        X_train,
        y_train,
        random_state=params["random_state"]
    )

    # Create data path and prevent error
    dirs = [
        params["train_data_path"],
        params["validation_data_path"],
        params["test_data_path"]
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Save original, resampled, and validation splits to their respective .npz files.
    save_npz(
        X_train,
        y_train,
        os.path.join(params["train_data_path"], "train_original.npz")
    )
    save_npz(
        X_train_resampled,
        y_train_resampled,
        os.path.join(params["train_data_path"], "train_resampled.npz")
    )
    save_npz(
        X_valid,
        y_valid,
        os.path.join(params["validation_data_path"], "valid.npz")
    )

    # Save X_test as JSON (input features)
    X_test.to_json(
        os.path.join(params["test_data_path"], "X_test.json"),
        orient="records",
        lines=True,
    )

    # Save y_test as JSON (true labels)
    y_test.to_frame(name=params["target_column"]).to_json(
        os.path.join(params["test_data_path"], "y_test.json"),
        orient="records",
        lines=True,
    )


if __name__ == "__main__":
    main()
