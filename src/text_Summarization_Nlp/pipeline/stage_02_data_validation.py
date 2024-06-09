from text_Summarization_Nlp.config.configuration import ConfigurationManager
from text_Summarization_Nlp.components.data_validation import DataValidation
from text_Summarization_Nlp.logging import logger


class DataValidationTrainingPipeline:
    def __init__(self):
        """
        Initialize the DataValidationTrainingPipeline with a configuration manager.
        """
        try:
            self.config_manager = ConfigurationManager()
            logger.info("ConfigurationManager initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ConfigurationManager: {e}")
            raise e

    def run_data_validation(self):
        """
        Run the data validation process to ensure all required files exist.
        """
        try:
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.validate_all_files_exist()
            logger.info("Data validation process completed successfully.")
        except Exception as e:
            logger.error(f"Data validation process failed: {e}")
            raise e

    def main(self):
        """
        Execute the main pipeline which runs the data validation process.
        """
        try:
            self.run_data_validation()
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise e

# if __name__ == "__main__":
#     try:
#         pipeline = DataValidationTrainingPipeline()
#         pipeline.main()
#     except Exception as e:
#         logger.error(f"Failed to run the DataValidationTrainingPipeline: {e}")
