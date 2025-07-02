import os
import argparse
import logging
import subprocess
from urllib.parse import urlparse


def download_zip(url: str, download_dir: str = "data/raw") -> str:
    """
    Downloads a ZIP file from a specified URL and saves it to the target directory.

    If the filename cannot be derived from the URL, a default name 'dataset.zip' is used.

    Args:
        url (str): URL pointing to the ZIP file to download.
        download_dir (str): Directory where the ZIP file will be saved. Defaults to "data/raw".

    Returns:
        str: Full path to the downloaded ZIP file.

    Raises:
        subprocess.CalledProcessError: If the download command fails.
    """
    os.makedirs(download_dir, exist_ok=True)

    # Extract name or fallback
    base = os.path.basename(urlparse(url).path)
    filename = f"{base}.zip" if base and not base.endswith(".zip") else (base or "dataset.zip")

    zip_path = os.path.join(download_dir, filename)

    logging.info(f"📥 Downloading ZIP from: {url}")
    cmd = ["curl", "-L", "-o", zip_path, url]
    subprocess.run(cmd, check=True)
    logging.info(f"✅ Download complete → {zip_path}")

    return zip_path


def extract_zip(zip_path: str, output_dir: str = "data/raw") -> None:
    """
    Extracts the contents of a ZIP file to the specified directory.

    Args:
        zip_path (str): Path to the ZIP file to extract.
        output_dir (str): Target directory for the extracted contents.
                          Defaults to "data/raw".

    Raises:
        subprocess.CalledProcessError: If the unzip command fails.
    """
    logging.info(f"📂 Extracting {zip_path} to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["unzip", "-o", zip_path, "-d", output_dir], check=True)
    logging.info("✅ Extraction complete.")
    subprocess.run(["rm", "-f", zip_path], check=True)


if __name__ == "__main__":
    """
    Entry point for command-line execution. Parses arguments and performs
    download and extraction of a ZIP file from a given URL.

    Example:
        python collect_and_extract.py "https://example.com/data.zip" --out data/raw
    """
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="Download and extract zip file from URL.")
    parser.add_argument("url", help="Link to a .zip file")
    parser.add_argument("--out", default="data/raw", help="Directory to extract to (default: data/raw)")
    args = parser.parse_args()

    zip_path = download_zip(args.url, args.out)
    extract_zip(zip_path, args.out)
