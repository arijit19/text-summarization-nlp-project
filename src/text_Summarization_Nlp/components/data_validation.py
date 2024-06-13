import os
from text_Summarization_Nlp.logging import logger
from text_Summarization_Nlp.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        """
        Initialize the DataValidation class with the provided configuration.
        
        Args:
            config (DataValidationConfig): Configuration for data validation.
        """
        self.config = config

    def validate_all_files_exist(self) -> bool:
        """
        Validate that all required files exist in the specified directory.

        Returns:
            bool: True if all required files exist, False otherwise.

        Raises:
            Exception: If an error occurs during validation.
        """
        try:
            required_files = set(self.config.ALL_REQUIRED_FILES)
            existing_files = set(os.listdir(self.config.REQUIRED_DIR))

            missing_files = required_files - existing_files
            validation_status = not missing_files

            status_message = f"Validation status: {validation_status}. "
            if missing_files:
                status_message += f"Missing files: {', '.join(missing_files)}"
                logger.error(status_message)
            else:
                status_message += "All required files are present."
                logger.info(status_message)

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(status_message)

            return validation_status
        
        except Exception as e:
            logger.error(f"An error occurred during file validation: {e}")
            raise e

