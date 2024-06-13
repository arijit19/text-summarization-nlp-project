import os
from box.exceptions import BoxValueError
import yaml
from text_Summarization_Nlp.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, List

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: For any other exceptions.

    Returns:
        ConfigBox: The content of the YAML file as a ConfigBox.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                logger.error(f"YAML file at {path_to_yaml} is empty.")
                raise ValueError("YAML file is empty")
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        logger.error(f"Error reading YAML file at {path_to_yaml}: {e}")
        raise ValueError("YAML file is empty") from e
    except Exception as e:
        logger.error(f"An error occurred while reading YAML file at {path_to_yaml}: {e}")
        raise e

# @ensure_annotations
def create_directories(path_to_directories: List[Path], verbose: bool = True):
    """
    Create a list of directories.

    Args:
        path_to_directories (List[Path]): List of paths of directories to create.
        verbose (bool, optional): Enable verbose logging. Defaults to True.
    """
    for path in path_to_directories:
        try:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory at {path}: {e}")
            raise e

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get the size of a file in KB.

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size of the file in KB.
    """
    try:
        size_in_kb = round(os.path.getsize(path) / 1024)
        logger.info(f"Size of {path}: ~ {size_in_kb} KB")
        return f"~ {size_in_kb} KB"
    except Exception as e:
        logger.error(f"Failed to get size for {path}: {e}")
        raise e

