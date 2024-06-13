from text_Summarization_Nlp.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from text_Summarization_Nlp.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from text_Summarization_Nlp.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from text_Summarization_Nlp.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from text_Summarization_Nlp.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from text_Summarization_Nlp.logging import logger

def run_pipeline_stage(stage_name, pipeline_class):
    """
    Runs a specified pipeline stage.
    
    Args:
        stage_name (str): Name of the stage.
        pipeline_class (class): The pipeline class to run.
    """
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline_instance = pipeline_class()
        pipeline_instance.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"Exception occurred in stage {stage_name}: {e}")
        raise e

if __name__ == "__main__":
    stages = [
        ("Data Ingestion stage", DataIngestionTrainingPipeline),
        ("Data Validation stage", DataValidationTrainingPipeline),
        ("Data Transformation stage", DataTransformationTrainingPipeline),
        ("Model Trainer stage", ModelTrainerTrainingPipeline),
        ("Model Evaluation stage", ModelEvaluationTrainingPipeline)
    ]

    for stage_name, pipeline_class in stages:
        run_pipeline_stage(stage_name, pipeline_class)
