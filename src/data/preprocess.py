import pandas as pd
from sklearn.preprocessing import RobustScaler
import yaml
import os


def load_params(config_path: str = "params.yaml") -> dict:
    """
    Load preprocessing parameters from a YAML configuration file.

    Parameters:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Dictionary containing preprocessing parameters.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)["preprocess"]


def read_dataset(filename: str) -> pd.DataFrame:
    """
    Read a CSV file into a Pandas DataFrame.

    Parameters:
        filename (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    return pd.read_csv(filename)


def scale_columns(df: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
    """
    Scale specified columns in the DataFrame using RobustScaler.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns_to_scale (list): List of column names to scale.

    Returns:
        pd.DataFrame: DataFrame with scaled columns.
    """
    df_scaled = df.copy()
    scaler = RobustScaler()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df_scaled


def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    Save the DataFrame as a CSV file at the specified location.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): File path to store the CSV output.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    """
    Main function to execute preprocessing pipeline.
    """
    params = load_params()
    df = read_dataset(params["raw_data_path"])
    df_scaled = scale_columns(df, params["scale_columns"])
    save_processed_data(df_scaled, params["processed_data_path"])


if __name__ == "__main__":
    main()
