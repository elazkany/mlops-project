# tests/data/test_collect_and_extract.py

from unittest import mock
from src.data.collect_and_extract import download_zip, extract_zip


@mock.patch("subprocess.run")
def test_download_zip_creates_expected_path(mock_subprocess, tmp_path):
    """
    Tests that `download_zip` constructs the correct output path and calls subprocess.

    Mocks:
        - subprocess.run to avoid real network calls.

    Asserts:
        - Returned zip path ends with expected filename.
        - subprocess.run was called with the correct curl command.
    """
    test_url = "https://example.com/test.zip"
    out_dir = tmp_path / "downloads"

    zip_path = download_zip(test_url, str(out_dir))

    expected_path = out_dir / "test.zip"
    assert zip_path == str(expected_path)
    mock_subprocess.assert_called_with(["curl", "-L", "-o", str(expected_path), test_url], check=True)


@mock.patch("subprocess.run")
def test_extract_zip_unzips_and_removes(mock_subprocess, tmp_path):
    """
    Tests that `extract_zip` runs unzip and file removal commands using subprocess.

    Mocks:
        - subprocess.run to simulate successful zip extraction and cleanup.

    Asserts:
        - subprocess.run is called with the expected commands for unzip and rm.
    """
    zip_file = tmp_path / "data.zip"
    zip_file.touch()  # Create a fake zip file
    output_dir = tmp_path / "extracted"

    extract_zip(str(zip_file), str(output_dir))

    expected_unzip = ["unzip", "-o", str(zip_file), "-d", str(output_dir)]
    expected_rm = ["rm", "-f", str(zip_file)]

    mock_subprocess.assert_any_call(expected_unzip, check=True)
    mock_subprocess.assert_any_call(expected_rm, check=True)
