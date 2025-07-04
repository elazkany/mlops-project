import os
import tempfile
from unittest import mock
from src.data import check_for_new_data


def test_fetch_last_modified_returns_header():
    """
    Test that fetch_last_modified() successfully retrieves
    the 'Last-Modified' HTTP header from a mocked URL.
    """
    with mock.patch("requests.head") as mock_head:
        mock_response = mock.Mock()
        mock_response.headers = {"Last-Modified": "Wed, 03 Jul 2024 17:05:00 GMT"}
        mock_head.return_value = mock_response

        result = check_for_new_data.fetch_last_modified("https://example.com")
        assert result == "Wed, 03 Jul 2024 17:05:00 GMT"
        mock_head.assert_called_once()


def test_write_last_modified_timestamp_creates_file_with_content():
    """
    Test that write_last_modified_timestamp() correctly writes
    a given timestamp string to a file, creating directories as needed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "meta/last_modified.txt")
        timestamp = "Mon, 01 Jan 2024 12:00:00 GMT"

        check_for_new_data.write_last_modified_timestamp(file_path, timestamp)

        assert os.path.exists(file_path)
        with open(file_path) as f:
            content = f.read()
            assert content == timestamp


def test_main_logic_with_mocked_dependencies():
    """
    Test the full main() execution flow with mocked dependencies:
    - load_params() returns a test configuration
    - fetch_last_modified() returns a simulated timestamp
    Verifies that the correct timestamp is written to the file.
    """
    mock_params = {
        "url": "https://example.com",
        "last_updated": "temp/test_last_modified.txt",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        full_path = os.path.join(tmpdir, "test_last_modified.txt")
        mock_params["last_updated"] = full_path

        with mock.patch("src.data.check_for_new_data.load_params", return_value=mock_params), \
             mock.patch("src.data.check_for_new_data.fetch_last_modified", return_value="Thu, 04 Jul 2025 10:30:00 GMT"):

            check_for_new_data.main()

            assert os.path.exists(full_path)
            with open(full_path) as f:
                content = f.read()
                assert content == "Thu, 04 Jul 2025 10:30:00 GMT"
