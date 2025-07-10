# tests/test_run_train.py

import unittest
from unittest.mock import patch
import numpy as np


class TestRunTrain(unittest.TestCase):

    @patch("src.train.run_train.create_and_train_model")
    @patch("src.train.run_train.mlflow.set_tracking_uri")
    @patch("src.train.run_train.Path.mkdir")
    @patch("src.train.run_train.load_npz")
    @patch("src.train.run_train.load_params")
    def test_main_runs_all_models(
        self,
        mock_load_params,
        mock_load_npz,
        mock_mkdir,
        mock_set_tracking_uri,
        mock_create_and_train_model
    ):
        from src.train.run_train import main

        # === Mock return values ===
        mock_load_params.return_value = {"experiment_name": "test-experiment"}

        # Each call to load_npz returns a tuple (X, y)
        X_fake = np.random.rand(10, 5)
        y_fake = np.random.randint(0, 2, 10)
        mock_load_npz.side_effect = [
            (X_fake, y_fake),     # train_original
            (X_fake, y_fake),     # train_resampled
            (X_fake, y_fake)      # validation
        ]

        # === Run the main function ===
        main()

        # === Expectations ===
        self.assertEqual(mock_create_and_train_model.call_count, 4)

        expected_calls = [
            ("logistic_regression", "lr_without_smote"),
            ("logistic_regression", "lr_with_smote"),
            ("random_forest", "rf_without_smote"),
            ("random_forest", "rf_with_smote")
        ]

        for call_args, expected in zip(mock_create_and_train_model.call_args_list, expected_calls):
            args, kwargs = call_args
            self.assertEqual(kwargs["model_type"], expected[0])
            self.assertEqual(kwargs["model_name"], expected[1])
            self.assertTrue("experiment_name" in kwargs)
            self.assertIn("X_train", kwargs)
            self.assertIn("y_train", kwargs)
            self.assertIn("X_valid", kwargs)
            self.assertIn("y_valid", kwargs)

        mock_mkdir.assert_called_once()
        mock_set_tracking_uri.assert_called_once()


if __name__ == "__main__":
    unittest.main()
