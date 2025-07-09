# src/train/generate_cml_report.py

from PIL import Image
import json
import shutil
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from src.utils.io_load import load_params


def main():
    # Load experiment name from params.yaml
    experiment_name = load_params("experiment_tracking")["experiment_name"]

    # Compare runs: metrics + confusion matrices
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs([experiment.experiment_id])

    metrics_dict = {}
    confusion_matrices = []

    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        metrics_dict[run_name] = run.data.metrics

        # Download confusion_matrix.png
        try:
            cm_bytes = client.download_artifacts(run.info.run_id, "confusion_matrix.png")
            image = Image.open(cm_bytes)
            confusion_matrices.append((run_name, image))
        except Exception as e:
            print(f"Could not retrieve confusion matrix for {run_name}: {e}")

    # Save metrics as JSON
    with open("reports/compare_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # Plot all confusion matrices in a grid
    if confusion_matrices:
        num = len(confusion_matrices)
        cols = 2
        rows = (num + 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axs = axs.flatten() if num > 1 else [axs]

        for ax, (title, img) in zip(axs, confusion_matrices):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Hide any unused subplots
        for ax in axs[len(confusion_matrices):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("reports/compare_confusion_matrices.png")
        plt.close()

    # Load best model info
    with open("deployment/model_version.json") as f:
        best_info = json.load(f)

    best_run = client.get_run(best_info["run_id"])
    best_metrics = best_run.data.metrics

    # Save best model metrics to JSON for CML report
    with open("reports/best_model_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=4)

    # Download confusion_matrix.png artifact for CML report
    try:
        local_cm_path = client.download_artifacts(best_info["run_id"], "confusion_matrix.png")
        shutil.copy(local_cm_path, "reports/best_model_confusion_matrix.png")
        print("Saved best model confusion matrix as best_model_confusion_matrix.png")
    except Exception as e:
        print(f"Could not download confusion matrix artifact: {e}")


if __name__ == "__main__":
    main()
