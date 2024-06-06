import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format= '[%(asctime)s] : %(message)s')

project_name = 'text_Summarization_Nlp'

# List of files to create
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, fileName = os.path.split(filepath)

    try:
        # Create directory if it does not exist
        if filedir:
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Created directory {filedir} for the file {fileName}")

        # Create file if it does not exist or is empty
        if not filepath.exists() or filepath.stat().st_size == 0:
            filepath.touch(exist_ok=True)
            logging.info(f"Created empty file: {filepath}")
        else:
            logging.info(f"File {filepath} already exists and is not empty")
    
    except Exception as e:
        logging.error(f"Failed to create {filepath}. Reason: {e}")


