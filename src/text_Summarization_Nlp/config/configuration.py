import os
from typing import Optional
from pathlib import Path
from text_Summarization_Nlp.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from text_Summarization_Nlp.utils.common import read_yaml, create_directories
from text_Summarization_Nlp.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from text_Summarization_Nlp.logging import logger


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Optional[str] = CONFIG_FILE_PATH,
        params_filepath: Optional[str] = PARAMS_FILE_PATH
    ):
        try:
            self.config = read_yaml(Path(config_filepath))
            self.params = read_yaml(Path(params_filepath))
            create_directories([self.config.artifacts_root])
            logger.info("Configuration files loaded successfully and directories created.")
        except Exception as e:
            logger.error(f"Error initializing ConfigurationManager: {e}")
            raise e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config.data_ingestion
            create_directories([config.root_dir])
            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                source_URL=config.source_URL,
                local_data_file=config.local_data_file,
                unzip_dir=config.unzip_dir
            )
            logger.info("DataIngestionConfig created successfully.")
            return data_ingestion_config
        except Exception as e:
            logger.error(f"Error creating DataIngestionConfig: {e}")
            raise e

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            config = self.config.data_validation
            create_directories([config.root_dir])
            data_validation_config = DataValidationConfig(
                root_dir=config.root_dir,
                STATUS_FILE=config.STATUS_FILE,
                ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
            )
            logger.info("DataValidationConfig created successfully.")
            return data_validation_config
        except Exception as e:
            logger.error(f"Error creating DataValidationConfig: {e}")
            raise e

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            config = self.config.data_transformation
            create_directories([config.root_dir])
            data_transformation_config = DataTransformationConfig(
                root_dir=config.root_dir,
                data_path=config.data_path,
                tokenizer_name=config.tokenizer_name
            )
            logger.info("DataTransformationConfig created successfully.")
            return data_transformation_config
        except Exception as e:
            logger.error(f"Error creating DataTransformationConfig: {e}")
            raise e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            config = self.config.model_trainer
            params = self.params.TrainingArguments
            create_directories([config.root_dir])
            model_trainer_config = ModelTrainerConfig(
                root_dir=config.root_dir,
                data_path=config.data_path,
                model_ckpt=config.model_ckpt,
                num_train_epochs=params.num_train_epochs,
                warmup_steps=params.warmup_steps,
                per_device_train_batch_size=params.per_device_train_batch_size,
                weight_decay=params.weight_decay,
                logging_steps=params.logging_steps,
                evaluation_strategy=params.evaluation_strategy,
                eval_steps=params.eval_steps,
                save_steps=params.save_steps,
                gradient_accumulation_steps=params.gradient_accumulation_steps
            )
            logger.info("ModelTrainerConfig created successfully.")
            return model_trainer_config
        except Exception as e:
            logger.error(f"Error creating ModelTrainerConfig: {e}")
            raise e

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            config = self.config.model_evaluation
            create_directories([config.root_dir])
            model_evaluation_config = ModelEvaluationConfig(
                root_dir=config.root_dir,
                data_path=config.data_path,
                model_path=config.model_path,
                tokenizer_path=config.tokenizer_path,
                metric_file_name=config.metric_file_name
            )
            logger.info("ModelEvaluationConfig created successfully.")
            return model_evaluation_config
        except Exception as e:
            logger.error(f"Error creating ModelEvaluationConfig: {e}")
            raise e
