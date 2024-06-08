from text_Summarization_Nlp.config.configuration import ConfigurationManager
from text_Summarization_Nlp.components.data_ingestion import DataIngestion
from text_Summarization_Nlp.logging import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        """
        Initialize the DataIngestionTrainingPipeline with a configuration manager.
        """
        try:
            self.config_manager = ConfigurationManager()
            logger.info("ConfigurationManager initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ConfigurationManager: {e}")
            raise e

    def run_data_ingestion(self):
        """
        Run the data ingestion process which includes downloading and extracting files.
        """
        try:
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            logger.info("Data ingestion process completed successfully.")
        except Exception as e:
            logger.error(f"Data ingestion process failed: {e}")
            raise e

    def main(self):
        """
        Execute the main pipeline which runs the data ingestion process.
        """
        try:
            self.run_data_ingestion()
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise e

# if __name__ == "__main__":
#     try:
#         pipeline = DataIngestionTrainingPipeline()
#         pipeline.main()
#     except Exception as e:
#         logger.error(f"Failed to run the DataIngestionTrainingPipeline: {e}")
