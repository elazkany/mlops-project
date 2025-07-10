import unittest
from unittest.mock import patch, MagicMock, mock_open


class TestGenerateCMLReport(unittest.TestCase):

    @patch("src.train.generate_cml_report.shutil.copy")
    @patch("src.train.generate_cml_report.json.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.train.generate_cml_report.Image.open")
    @patch("src.train.generate_cml_report.plt")
    @patch("src.train.generate_cml_report.json.dump")
    @patch("src.train.generate_cml_report.MlflowClient")
    @patch("src.train.generate_cml_report.load_params")
    def test_main_creates_report(
        self,
        mock_load_params,
        mock_mlflow_client_class,
        mock_json_dump,
        mock_plt,
        mock_image_open,
        mock_file_open,
        mock_json_load,
        mock_shutil_copy
    ):
        from src.train.generate_cml_report import main

        # === Step 1: Load params and best model info ===
        mock_load_params.return_value = {"experiment_name": "test-experiment"}
        mock_json_load.return_value = {"run_id": "best_run_id"}

        # === Step 2: MLflow experiment and runs ===
        mock_client = MagicMock()
        mock_mlflow_client_class.return_value = mock_client

        experiment = MagicMock()
        experiment.experiment_id = "exp123"
        mock_client.get_experiment_by_name.return_value = experiment

        run1 = MagicMock()
        run1.info.run_id = "run_1"
        run1.data.metrics = {"accuracy": 0.91}
        run1.data.tags = {"mlflow.runName": "Model A"}

        run2 = MagicMock()
        run2.info.run_id = "run_2"
        run2.data.metrics = {"accuracy": 0.87}
        run2.data.tags = {"mlflow.runName": "Model B"}

        mock_client.search_runs.return_value = [run1, run2]

        # === Step 3: Image and artifact mocks ===
        mock_image1 = MagicMock(name="Image1")
        mock_image2 = MagicMock(name="Image2")
        mock_image_open.side_effect = [mock_image1, mock_image2]
        mock_client.download_artifacts.side_effect = [
            "cm1.png",  # for run_1
            "cm2.png",  # for run_2
            "best_cm.png"  # for best model
        ]

        # === Step 4: Best model metrics ===
        best_run = MagicMock()
        best_run.data.metrics = {"precision": 0.88, "recall": 0.81}
        mock_client.get_run.return_value = best_run

        # === Step 5: Patch plt.subplots to return (fig, axs) with flatten() ===
        mock_fig = MagicMock()
        ax1 = MagicMock()
        ax2 = MagicMock()
        axs_mock = MagicMock()
        axs_mock.flatten.return_value = [ax1, ax2]
        mock_plt.subplots.return_value = (mock_fig, axs_mock)

        # === Run the script ===
        main()

        # === Assertions ===
        self.assertEqual(mock_client.download_artifacts.call_count, 3)
        self.assertEqual(mock_image_open.call_count, 2)
        self.assertEqual(mock_json_dump.call_count, 2)
        mock_plt.savefig.assert_called_once_with("reports/compare_confusion_matrices.png")
        mock_shutil_copy.assert_called_once_with("best_cm.png", "reports/best_model_confusion_matrix.png")


if __name__ == "__main__":
    unittest.main()
