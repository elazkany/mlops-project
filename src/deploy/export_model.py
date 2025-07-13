# src/deply/export_model.py

import mlflow
from mlflow.tracking import MlflowClient
import json
from pathlib import Path
import shutil


def download_and_prepare_model():
    """
    Exports a trained MLflow model from the registry to a local folder for FastAPI deployment.

    Steps:
    - Load model metadata from `deployment/model_version.json`
    - Use MLflow client to get the download URI for the registered model
    - Copy all model artifact files into `deployment/model_artifacts`
    - Save the metadata alongside the model

    Raises:
        FileNotFoundError: If `model_version.json` does not exist
        MlflowException: If the model or version cannot be found
    """
    export_dir = Path("deployment/model_artifacts")

    # Set up MLflow tracking
    tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Load model name and version metadata
    version_file = Path("deployment/model_version.json")
    if not version_file.exists():
        raise FileNotFoundError("Model metadata file not found. Run the registration stage first.")

    with open(version_file, "r") as f:
        metadata = json.load(f)

    model_name = metadata["model_name"]
    version = metadata["version"]

    print(f"🔗 Fetching download URI for model '{model_name}' (version {version})...")
    download_uri = client.get_model_version_download_uri(name=model_name, version=version)
    source_dir = Path(download_uri.replace("file://", ""))

    if export_dir.exists():
        print(f"♻️ Overwriting existing artifacts at {export_dir}")
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    print(f"📦 Copying model files from {source_dir}")
    shutil.copytree(source_dir, export_dir, dirs_exist_ok=True)

    with open(export_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("✅ Model exported successfully to 'deployment/model_artifacts'!")


if __name__ == "__main__":
    download_and_prepare_model()
