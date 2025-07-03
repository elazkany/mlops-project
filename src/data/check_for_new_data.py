# src/data/check_for_new_data.py

import os
import yaml
import requests


def load_url_from_params(params_path="params.yaml"):
    """
    Loads the download URL and metadata file path from a YAML configuration file.

    Args:
        params_path (str): Path to the YAML file containing the download configuration.
                           Defaults to "params.yaml".

    Returns:
        tuple[str, str]: A tuple containing the download URL and the path to the metadata file
                         where the last modified timestamp should be stored.
    """
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
        return params["download"]["url"], params["download"]["last_updated"]


def fetch_last_modified(url):
    """
    Sends a HEAD request to a given URL and retrieves the Last-Modified header.

    Args:
        url (str): The URL of the file to inspect.

    Returns:
        str: The value of the 'Last-Modified' HTTP header, or an empty string if not found.
    """
    response = requests.head(url)
    return response.headers.get("Last-Modified", "")


def write_last_modified_timestamp(filepath, timestamp):
    """
    Writes the provided timestamp to the specified file. Creates directories if needed.

    Args:
        filepath (str): The destination path where the timestamp should be written.
        timestamp (str): The timestamp value to write (e.g., from Last-Modified HTTP header).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(timestamp)


def main():
    """
    Orchestrates the metadata refresh process:
    1. Loads configuration from params.yaml.
    2. Sends a request to fetch the Last-Modified timestamp of the target URL.
    3. Writes the retrieved timestamp to the metadata file.
    """
    url, meta_file = load_url_from_params()
    last_modified = fetch_last_modified(url)
    write_last_modified_timestamp(meta_file, last_modified)


if __name__ == "__main__":
    main()
