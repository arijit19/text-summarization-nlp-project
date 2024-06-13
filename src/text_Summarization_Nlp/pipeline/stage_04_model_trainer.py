from text_Summarization_Nlp.config.configuration import ConfigurationManager
from text_Summarization_Nlp.components.model_trainer import ModelTrainer
from text_Summarization_Nlp.logging import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        """
        Initialize the ModelTrainerTrainingPipeline with a configuration manager.
        """
        try:
            self.config_manager = ConfigurationManager()
            logger.info("ConfigurationManager initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ConfigurationManager: {e}")
            raise e

    def run_model_trainer(self):
        """
        Run the model training process.
        """
        try:
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
            logger.info("Model training process completed successfully.")
        except Exception as e:
            logger.error(f"Model training process failed: {e}")
            raise e

    def main(self):
        """
        Execute the main pipeline which runs the model training process.
        """
        try:
            self.run_model_trainer()
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise e
