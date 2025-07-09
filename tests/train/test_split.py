# tests/test_split.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.train.split import apply_smote


class TestSplitModule(unittest.TestCase):

    def test_apply_smote_balances_classes(self):
        # Imbalanced synthetic dataset
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series([0]*90 + [1]*10)

        X_res, y_res = apply_smote(X, y, random_state=42)

        # After SMOTE, class counts should be equal
        class_counts = pd.Series(y_res).value_counts()
        self.assertEqual(class_counts[0], class_counts[1])
        self.assertEqual(X_res.shape[0], y_res.shape[0])
        self.assertEqual(X_res.shape[1], X.shape[1])  # same number of features

    @patch("src.train.split.save_npz")
    @patch("src.train.split.os.makedirs")
    @patch("src.train.split.apply_smote")
    @patch("src.train.split.train_test_split")
    @patch("src.train.split.load_dataset")
    @patch("src.train.split.load_params")
    def test_main_function_flow(
        self,
        mock_load_params,
        mock_load_dataset,
        mock_train_test_split,
        mock_apply_smote,
        mock_makedirs,
        mock_save_npz
    ):
        from src.train.split import main

        # === Setup mocks ===
        mock_load_params.return_value = {
            "processed_data_path": "data/processed.csv",
            "target_column": "target",
            "test_size": 0.2,
            "random_state": 42,
            "train_data_path": "data/train",
            "validation_data_path": "data/valid",
            "test_data_path": "data/test"
        }

        df = pd.DataFrame({
            "feature1": range(10),
            "feature2": range(10, 20),
            "target": [0]*8 + [1]*2
        })
        mock_load_dataset.return_value = df

        # 3 splits = 2 calls to train_test_split
        mock_train_test_split.side_effect = [
            (df.iloc[:6, :-1], df.iloc[6:, :-1], df["target"][:6], df["target"][6:]),
            (df.iloc[:3, :-1], df.iloc[3:6, :-1], df["target"][:3], df["target"][3:6])
        ]

        # Mock SMOTE output
        mock_apply_smote.return_value = (
            pd.DataFrame(np.random.rand(6, 2)),
            pd.Series([0, 0, 1, 1, 0, 1])
        )

        # Run
        with patch("src.train.split.pd.DataFrame.to_json") as mock_to_json, \
             patch("src.train.split.pd.Series.to_frame") as mock_series_to_frame:

            mock_series_to_frame.return_value.to_json = MagicMock()
            main()

        # === Assertions ===
        self.assertTrue(mock_makedirs.called)
        self.assertTrue(mock_save_npz.called)
        self.assertTrue(mock_to_json.called)
        self.assertTrue(mock_series_to_frame.called)


if __name__ == "__main__":
    unittest.main()
