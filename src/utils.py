import yaml
from typing import Any


def load_config(file_path: str = "configs/config.yml") -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict[str, Any]: Parsed YAML content as a dictionary.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
