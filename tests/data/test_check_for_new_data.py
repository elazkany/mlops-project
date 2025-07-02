# tests/data/test_check_for_new_data.py

import os
import yaml
import pytest
from unittest import mock
from src.data.check_for_new_data import (
    load_url_from_params,
    fetch_last_modified,
    write_last_modified_timestamp
)


@pytest.fixture
def project_env(tmp_path, monkeypatch):
    """
    Creates an isolated test environment with a temporary 'params.yaml' file
    and a writable 'data' directory.

    - Sets the working directory to a temporary folder
    - Creates a mock download URL and output metadata path
    - Returns a dict with the expected test values for validation

    Args:
        tmp_path (pathlib.Path): Built-in pytest fixture for temporary file paths
        monkeypatch (function): Fixture for safely modifying environment behavior

    Returns:
        dict: A dictionary containing the 'url' and 'last_updated' path for testing
    """
    monkeypatch.chdir(tmp_path)
    params_yaml = {
        "download": {
            "url": "https://example.com/data.zip",
            "last_updated": "data/last_updated.txt"
        }
    }
    with open(tmp_path / "params.yaml", "w") as f:
        yaml.dump(params_yaml, f)

    os.makedirs(tmp_path / "data", exist_ok=True)
    return params_yaml["download"]


def test_load_url_from_params(project_env):
    """
    Tests whether the function correctly loads download metadata
    from a local 'params.yaml' file.

    Asserts:
        - Extracted URL matches expected test URL
        - Metadata file path matches expected location
    """
    url, meta_file = load_url_from_params()
    assert url == project_env["url"]
    assert meta_file == project_env["last_updated"]


@mock.patch("requests.head")
def test_fetch_last_modified(mock_head):
    """
    Tests that 'fetch_last_modified' properly extracts the 'Last-Modified'
    header from an HTTP response.

    Mocks:
        - requests.head: to simulate a successful response with a known header

    Asserts:
        - Returned timestamp matches the mocked header value
    """
    mock_head.return_value.headers = {
        "Last-Modified": "Wed, 02 Jul 2025 18:00:00 GMT"
    }

    result = fetch_last_modified("https://example.com/data.zip")
    assert result == "Wed, 02 Jul 2025 18:00:00 GMT"


def test_write_last_modified_timestamp(project_env):
    """
    Tests that the timestamp is correctly written to a file on disk.

    - Verifies the file is created
    - Confirms written content matches the test timestamp

    Asserts:
        - File exists
        - File content matches expected timestamp
    """
    test_timestamp = "Wed, 02 Jul 2025 18:00:00 GMT"
    file_path = project_env["last_updated"]

    write_last_modified_timestamp(file_path, test_timestamp)

    assert os.path.exists(file_path)
    with open(file_path) as f:
        assert f.read() == test_timestamp
