from text_Summarization_Nlp.config.configuration import ConfigurationManager
from text_Summarization_Nlp.components.data_transformation import DataTransformation
from text_Summarization_Nlp.logging import logger

class DataTransformationTrainingPipeline:
    def __init__(self):
        """
        Initialize the DataTransformationTrainingPipeline with a configuration manager.
        """
        try:
            self.config_manager = ConfigurationManager()
            logger.info("ConfigurationManager initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ConfigurationManager: {e}")
            raise e

    def run_data_transformation(self):
        """
        Run the data transformation process.
        """
        try:
            data_transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.convert()
            logger.info("Data transformation process completed successfully.")
        except Exception as e:
            logger.error(f"Data transformation process failed: {e}")
            raise e

    def main(self):
        """
        Execute the main pipeline which runs the data transformation process.
        """
        try:
            self.run_data_transformation()
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise e


