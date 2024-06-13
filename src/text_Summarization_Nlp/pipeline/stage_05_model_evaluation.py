from text_Summarization_Nlp.config.configuration import ConfigurationManager
from text_Summarization_Nlp.components.model_evaluation import ModelEvaluation
from text_Summarization_Nlp.logging import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        """
        Initialize the ModelEvaluationTrainingPipeline with a configuration manager.
        """
        try:
            self.config_manager = ConfigurationManager()
            logger.info("ConfigurationManager initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ConfigurationManager: {e}")
            raise e

    def run_model_evaluation(self):
        """
        Run the model evaluation process.
        """
        try:
            model_evaluation_config = self.config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.evaluate()
            logger.info("Model evaluation process completed successfully.")
        except Exception as e:
            logger.error(f"Model evaluation process failed: {e}")
            raise e

    def main(self):
        """
        Execute the main pipeline which runs the model evaluation process.
        """
        try:
            self.run_model_evaluation()
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise e

