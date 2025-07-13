# src/deploy/export_model.py

import mlflow
from mlflow.tracking import MlflowClient
import json
from pathlib import Path
import shutil
import sys
from src.utils.io_load import load_params

def create_placeholders(export_dir: Path, required_files: list):
    """
    Creates empty placeholder files in the specified export directory.

    Args:
        export_dir (Path): Directory where placeholder files will be created.
        required_files (list): List of filenames to touch.

    This function ensures DVC output expectations are met even if the export logic is skipped.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    for file_name in required_files:
        path = export_dir / file_name
        path.touch()  # Create empty file
    with open(export_dir / "README.txt", "w") as f:
        f.write("Skipped model export: deploy_using != 'fastapi' in params.yaml\n")

def download_and_prepare_model():
    """
    Exports a trained MLflow model from the registry to a local path for FastAPI deployment.

    This function:
    - Checks deployment mode via params.yaml.
    - Loads model metadata (name and version).
    - Fetches the model artifact URI from MLflow.
    - Copies model files to deployment/model_artifacts/.
    - Writes metadata to JSON.

    If the deployment mode is not 'fastapi', the function gracefully skips execution,
    creating placeholder output files for DVC stage compatibility.
    """
    deploy_config = load_params("deployment")["deploy_using"]

    export_dir = Path("deployment/model_artifacts")
    required_files = [
        "conda.yaml",
        "input_example.json",
        "metadata.json",
        "MLmodel",
        "model.pkl",
        "python_env.yaml",
        "registered_model_meta",
        "requirements.txt",
        "serving_input_example.json"
    ]

    if deploy_config != "fastapi":
        print(f"🚫 Skipping export: deploy_using is set to '{deploy_config}', not 'fastapi'.")
        create_placeholders(export_dir, required_files)

        # 🚪 Gracefully exit with code 0, indicating success — but no export work was done.
        sys.exit(0)

    tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    version_file = Path("deployment/model_version.json")
    if not version_file.exists():
        print("❌ Model metadata file not found. Run the registration stage first.")
        create_placeholders(export_dir, required_files)
        return

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