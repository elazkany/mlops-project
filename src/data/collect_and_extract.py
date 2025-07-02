# src/data/collect_and_extract.py

import os
import logging
import subprocess
from urllib.parse import urlparse
from argparse import ArgumentParser


def download_zip(url: str, download_dir: str = "data/raw") -> str:
    """
    Downloads a ZIP file from a given URL and saves it to the specified directory.

    If the URL does not include a filename, 'dataset.zip' is used by default.

    Args:
        url (str): The direct URL to the ZIP file.
        download_dir (str): Directory where the ZIP file will be saved.

    Returns:
        str: Absolute path to the downloaded ZIP file.

    Raises:
        subprocess.CalledProcessError: If the curl command fails to complete.
    """
    os.makedirs(download_dir, exist_ok=True)

    base = os.path.basename(urlparse(url).path)
    filename = f"{base}.zip" if base and not base.endswith(".zip") else (base or "dataset.zip")
    zip_path = os.path.join(download_dir, filename)

    logging.info(f"📥 Downloading ZIP from: {url}")
    subprocess.run(["curl", "-L", "-o", zip_path, url], check=True)
    logging.info(f"✅ Download complete → {zip_path}")

    return zip_path


def extract_zip(zip_path: str, output_dir: str = "data/raw") -> None:
    """
    Extracts a ZIP file's contents into the specified directory and deletes the ZIP after extraction.

    Args:
        zip_path (str): Path to the downloaded ZIP file.
        output_dir (str): Destination directory for the extracted contents.

    Raises:
        subprocess.CalledProcessError: If the unzip or remove operation fails.
    """
    logging.info(f"📂 Extracting {zip_path} to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["unzip", "-o", zip_path, "-d", output_dir], check=True)
    logging.info("✅ Extraction complete.")
    subprocess.run(["rm", "-f", zip_path], check=True)


def main():
    """
    Parses command-line arguments and runs the full download and extraction workflow.
    """
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    parser = ArgumentParser(description="Download and extract a ZIP file from a given URL.")
    parser.add_argument("url", help="Direct link to a .zip file")
    parser.add_argument("--out", default="data/raw", help="Destination directory for the extracted files")
    args = parser.parse_args()

    zip_path = download_zip(args.url, args.out)
    extract_zip(zip_path, args.out)


if __name__ == "__main__":
    main()
