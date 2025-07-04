import pandas as pd
from sklearn.preprocessing import RobustScaler
from utils.io_load import load_params, load_dataset
from utils.io_save import save_dataframe


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


def main():
    """
    Main function to execute preprocessing pipeline.
    """
    params = load_params("preprocess")
    df = load_dataset(params["raw_data_path"])
    df_scaled = scale_columns(df, params["scale_columns"])
    save_dataframe(df_scaled, params["processed_data_path"])


if __name__ == "__main__":
    main()
