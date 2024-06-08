import os
import urllib.request as request
import zipfile
from text_Summarization_Nlp.logging import logger
from text_Summarization_Nlp.utils.common import get_size
from pathlib import Path
from text_Summarization_Nlp.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Downloads the file from the specified URL if it does not already exist locally.

        Raises:
            Exception: If there is an error during the download.
        """
        try:
            if not self.config.local_data_file.exists():
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_data_file
                )
                logger.info(f"{filename} downloaded with following info: \n{headers}")
            else:
                logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
        except Exception as e:
            logger.error(f"Failed to download file from {self.config.source_URL}: {e}")
            raise e

    def extract_zip_file(self):
        """
        Extracts the zip file into the specified directory.

        Raises:
            Exception: If there is an error during the extraction.
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extracted {self.config.local_data_file} to {unzip_path}")
        except zipfile.BadZipFile as e:
            logger.error(f"Bad zip file {self.config.local_data_file}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Failed to extract zip file {self.config.local_data_file}: {e}")
            raise e
