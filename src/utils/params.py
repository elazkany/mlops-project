# src/utils/params.py

import yaml


def load_params(section: str, path: str = "params.yaml") -> dict:
    """
    Load a specific section from the params.yaml config file.

    Parameters:
        section (str): Top-level section name in the YAML.
        path (str): Path to the YAML file.

    Returns:
        dict: Dictionary containing the requested section’s parameters.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config.get(section, {})
