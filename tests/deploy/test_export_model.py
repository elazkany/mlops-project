import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.deploy.export_model import download_and_prepare_model


@pytest.fixture
def temp_registry(tmp_path):
    """
    Sets up a fake MLflow registry artifact folder with a dummy model file.
    Also writes a dummy model_version.json for metadata.
    """
    # Create fake artifact directory
    artifact_dir = tmp_path / "mlruns" / "123" / "models" / "fake_model_path" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "model.pkl").write_text("pretend model bytes here")

    # Create deployment/model_version.json file
    metadata = {
        "model_name": "fake_model",
        "version": "1"
    }
    version_file = tmp_path / "deployment" / "model_version.json"
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text(json.dumps(metadata))

    return {
        "registry_dir": artifact_dir,
        "metadata": metadata,
        "version_file": version_file,
        "export_dir": tmp_path / "deployment" / "model_artifacts",
    }


@patch("src.deploy.export_model.mlflow.set_tracking_uri")
@patch("src.deploy.export_model.MlflowClient")
def test_download_and_prepare_model_creates_expected_outputs(
    mock_mlflow_client_class, mock_set_uri, temp_registry, monkeypatch
):
    """
    Ensures that the export_model script copies model artifacts and writes metadata.json
    when run against a mocked MLflow registry and metadata file.
    """
    # Mock get_model_version_download_uri to point to the fake artifact folder
    mock_client = MagicMock()
    uri = f"file://{temp_registry['registry_dir']}"
    mock_client.get_model_version_download_uri.return_value = uri
    mock_mlflow_client_class.return_value = mock_client

    # Monkeypatch Path to reroute all file access to tmp_path
    real_path_class = Path

    def patched_path(p):
        return real_path_class(temp_registry['version_file']).parent.parent / p

    monkeypatch.setattr("src.deploy.export_model.Path", patched_path)

    # Run export
    download_and_prepare_model()

    # Assertions
    export_dir = temp_registry["export_dir"]
    assert (export_dir / "model.pkl").exists()
    assert (export_dir / "metadata.json").exists()

    with open(export_dir / "metadata.json") as f:
        saved_metadata = json.load(f)
        assert saved_metadata == temp_registry["metadata"]
