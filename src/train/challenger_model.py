# src/train/challenger_model.py

from pathlib import Path
import json
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from src.utils.io_load import load_params


def select_and_register_best_model(
        experiment_name,
        precision_threshold,
        model_registry_name,
        ):

    # Set tracking URI
    mlflow.set_tracking_uri(f"file://{Path('mlruns').resolve()}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    runs = client.search_runs([experiment.experiment_id])

    best_run = None
    best_recall = -1

    # Loop through each MLflow run retrieved from the experiment.
    for run in runs:
        # Extract the dictionary of all logged metrics for the current run
        metrics = run.data.metrics

        # Gets the human-readable name of the run. If not available, uses the run ID.
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)

        precision = metrics.get("precision")
        recall = metrics.get("recall")

        if precision is None or recall is None:
            print(f"Skipping {run_name} — missing precision or recall.")
            continue

        if precision >= precision_threshold and recall > best_recall:
            best_recall = recall
            best_run = run

    if best_run:
        run_id = best_run.info.run_id
        print(f"Best model found: {best_run.data.tags.get('mlflow.runName', run_id)}")
        print(f"  Precision: {best_run.data.metrics['precision']}")
        print(f"  Recall: {best_run.data.metrics['recall']}")

        try:
            result = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=model_registry_name
            )
            print(f"Model registered as '{result.name}', version {result.version}")

            # Promote to Production
            client.transition_model_version_stage(
                name=model_registry_name,
                version=result.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"Model version {result.version} promoted to 'Production'.")

            # Store the model version for downstream use
            with open("deployment/model_version.json", "w") as f:
                json.dump({
                    "model_name": result.name,
                    "version": result.version,
                    "run_id": run_id
                }, f, indent=4)

        except MlflowException as e:
            print(f"Failed to register model: {e}")
    else:
        print("No suitable model found that meets the precision threshold.")


def main():
    experiment_name = load_params("experiment_tracking")["experiment_name"]
    precision_threshold = load_params("experiment_tracking")["precision_threshold"]
    best_model_name = load_params("experiment_tracking")["best_model_name"]

    select_and_register_best_model(
        experiment_name=experiment_name,
        precision_threshold=precision_threshold,
        model_registry_name=best_model_name,
    )


if __name__ == "__main__":
    main()
