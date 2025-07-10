# tests/test_challenger_model.py

import unittest
from unittest.mock import patch, MagicMock, mock_open
from src.train.challenger_model import select_and_register_best_model


class TestChallengerModel(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("mlflow.register_model")
    @patch("src.train.challenger_model.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_select_and_register_best_model_success(
        self, mock_set_uri, mock_client_class, mock_register_model,
        mock_json_dump, mock_file
    ):
        # === Setup mocks ===
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock experiment and runs
        experiment = MagicMock()
        experiment.experiment_id = "1234"
        mock_client.get_experiment_by_name.return_value = experiment

        # Create two fake runs
        run1 = MagicMock()
        run1.data.metrics = {"precision": 0.88, "recall": 0.60}
        run1.data.tags = {"mlflow.runName": "Run 1"}
        run1.info.run_id = "run_1"

        run2 = MagicMock()
        run2.data.metrics = {"precision": 0.91, "recall": 0.75}
        run2.data.tags = {"mlflow.runName": "Run 2"}
        run2.info.run_id = "run_2"

        mock_client.search_runs.return_value = [run1, run2]

        # Mock model registration response
        result = MagicMock()
        result.name = "registered-model"
        result.version = 3
        mock_register_model.return_value = result

        # === Run ===
        select_and_register_best_model(
            experiment_name="my-experiment",
            precision_threshold=0.90,
            model_registry_name="MyBestModel"
        )

        # === Assert ===
        mock_set_uri.assert_called_once()
        mock_client.get_experiment_by_name.assert_called_once_with("my-experiment")
        mock_register_model.assert_called_once_with(
            model_uri="runs:/run_2/model",
            name="MyBestModel"
        )
        mock_json_dump.assert_called_once()
        mock_file.assert_called_once_with("deployment/model_version.json", "w")

    @patch("src.train.challenger_model.MlflowClient")
    def test_no_experiment_found(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        mock_client_class.return_value = mock_client

        with patch("builtins.print") as mock_print:
            select_and_register_best_model(
                experiment_name="not-there",
                precision_threshold=0.85,
                model_registry_name="dummy"
            )
            mock_print.assert_any_call("Experiment 'not-there' not found.")

    @patch("src.train.challenger_model.MlflowClient")
    def test_no_qualifying_run(self, mock_client_class):
        mock_client = MagicMock()
        experiment = MagicMock()
        experiment.experiment_id = "1234"
        mock_client.get_experiment_by_name.return_value = experiment

        run = MagicMock()
        run.data.metrics = {"precision": 0.5, "recall": 0.4}  # below threshold
        run.data.tags = {}
        run.info.run_id = "id"
        mock_client.search_runs.return_value = [run]
        mock_client_class.return_value = mock_client

        with patch("builtins.print") as mock_print:
            select_and_register_best_model(
                experiment_name="test",
                precision_threshold=0.9,
                model_registry_name="Nope"
            )
            mock_print.assert_any_call("No suitable model found that meets the precision threshold.")


if __name__ == "__main__":
    unittest.main()
