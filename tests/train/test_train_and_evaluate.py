"""
Test the training and evaluation pipeline using large, SMOTE-safe mock data.
"""

import numpy as np
from unittest import mock
from src.train import train_and_evaluate


@mock.patch("src.train.train_and_evaluate.load_params")
@mock.patch("src.train.train_and_evaluate.load_npz")
@mock.patch("src.train.train_and_evaluate.train_model")
@mock.patch("src.train.train_and_evaluate.evaluate_model")
@mock.patch("src.train.train_and_evaluate.save_metrics")
@mock.patch("src.train.train_and_evaluate.save_predictions")
@mock.patch("src.train.train_and_evaluate.save_roc_curve")
@mock.patch("src.train.train_and_evaluate.joblib.dump")
def test_main_pipeline_runs_without_errors(
    mock_joblib_dump,
    mock_save_roc,
    mock_save_preds,
    mock_save_metrics,
    mock_eval_model,
    mock_train_model,
    mock_load_npz,
    mock_load_params,
):
    """
    Test that train_and_evaluate.main() runs end-to-end using mocked I/O and large data.
    Ensures no failures due to class imbalance or SMOTE dependencies.
    """

    # Create large, balanced mock data to avoid SMOTE errors
    X_train = np.random.rand(100, 10)
    y_train = np.array([0]*50 + [1]*50)
    X_test = np.random.rand(20, 10)
    y_test = np.array([0, 1]*10)

    # Mocked function outputs
    mock_load_params.return_value = {"n_estimators": 100}
    mock_load_npz.side_effect = [(X_train, y_train), (X_test, y_test)]
    mock_train_model.return_value = "mock_model"
    mock_eval_model.return_value = (
        {"accuracy": 0.95},
        [0, 1]*10,
        [0.1, 0.9]*10,
    )

    # Run main pipeline
    train_and_evaluate.main()

    # Validate expected function calls
    mock_load_params.assert_called_once()
    assert mock_load_npz.call_count == 2
    mock_train_model.assert_called_once_with(X_train, y_train, {"n_estimators": 100})
    mock_eval_model.assert_called_once_with("mock_model", X_test, y_test)
    mock_joblib_dump.assert_called_once()
    mock_save_metrics.assert_called_once()
    mock_save_preds.assert_called_once()
    mock_save_roc.assert_called_once()
