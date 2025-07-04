# tests/utils/test_plot_utils.py

import pytest
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from src.utils.plot_utils import plot_confusion_matrix
matplotlib.use("Agg")  # Use non-GUI backend for tests


@pytest.fixture
def trained_model():
    """
    Provides a trained classifier and test set.
    """
    X, y = make_classification(n_samples=100, n_features=4, n_informative=2,
                               n_redundant=0, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X[:80], y[:80])
    return model, X[80:], y[80:]


def test_plot_confusion_matrix(tmp_path, trained_model, monkeypatch):
    """
    Tests that the confusion matrix plot is generated and saved to file.
    """
    model, X_test, y_test = trained_model

    # Change working directory to temporary path
    monkeypatch.chdir(tmp_path)

    # Run the plotting function
    plot_confusion_matrix(model, X_test, y_test)

    # Check if file was saved
    plot_path = tmp_path / "confusion_matrix.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0  # Check that file is not empty
