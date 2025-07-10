# tests/test_train_and_evaluate.py

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.train.train_and_evaluate import create_and_train_model


class TestTrainAndEvaluate(unittest.TestCase):

    @patch("src.train.train_and_evaluate.mlflow")
    @patch("src.train.train_and_evaluate.os.remove")
    @patch("src.train.train_and_evaluate.plt")
    @patch("src.train.train_and_evaluate.ConfusionMatrixDisplay")
    @patch("src.train.train_and_evaluate.infer_signature")
    @patch("src.train.train_and_evaluate.evaluate_model")
    @patch("src.train.train_and_evaluate.train_model")
    @patch("src.train.train_and_evaluate.load_params")
    def test_create_and_train_model(
        self, mock_load_params, mock_train_model, mock_evaluate_model,
        mock_infer_signature, mock_conf_matrix, mock_plt,
        mock_os_remove, mock_mlflow
    ):
        # === Setup ===
        X_train = np.random.rand(10, 5)
        y_train = np.random.randint(0, 2, size=10)
        X_valid = np.random.rand(4, 5)
        y_valid = np.random.randint(0, 2, size=4)

        model_type = "random_forest"
        model_name = "rf-test"
        experiment_name = "unit-test-experiment"

        mock_model = MagicMock()
        mock_train_model.return_value = mock_model
        mock_model.predict.return_value = np.zeros(len(X_train))

        mock_evaluate_model.return_value = ({"accuracy": 0.99}, None, None)
        mock_load_params.return_value = {
            "random_forest": {"model_params": {"n_estimators": 10}}
        }
        mock_infer_signature.return_value = "fake-signature"
        mock_mlflow.active_run.return_value = True

        # === Run ===
        create_and_train_model(
            model_type, model_name,
            X_train, y_train,
            X_valid, y_valid,
            experiment_name
        )

        # === Assertions ===
        mock_train_model.assert_called_once()
        mock_evaluate_model.assert_called_once_with(mock_model, X_valid, y_valid)
        mock_infer_signature.assert_called_once()
        mock_mlflow.set_experiment.assert_called_with(experiment_name)
        mock_mlflow.start_run.assert_called()
        mock_mlflow.sklearn.log_model.assert_called()
        mock_os_remove.assert_called_with("confusion_matrix.png")
        mock_plt.savefig.assert_called_with("confusion_matrix.png")


if __name__ == "__main__":
    unittest.main()
