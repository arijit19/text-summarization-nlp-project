from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for data ingestion.

    Attributes:
        root_dir (Path): Directory for storing ingested data.
        source_URL (str): URL to download the data from.
        local_data_file (Path): Path to the local data file.
        unzip_dir (Path): Directory to unzip the data.
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration for data validation.

    Attributes:
        root_dir (Path): Directory for storing validation results.
        STATUS_FILE (str): File to store validation status.
        ALL_REQUIRED_FILES (List[str]): List of required files for validation.
    """
    root_dir: Path
    STATUS_FILE: str
    REQUIRED_DIR: Path
    ALL_REQUIRED_FILES: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for data transformation.

    Attributes:
        root_dir (Path): Directory for storing transformed data.
        data_path (Path): Path to the input data.
        tokenizer_name (str): Name of the tokenizer to use.
    """
    root_dir: Path
    data_path: Path
    tokenizer_name: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration for model training.

    Attributes:
        root_dir (Path): Directory for storing training artifacts.
        data_path (Path): Path to the training data.
        model_ckpt (Path): Path to the model checkpoint.
        num_train_epochs (int): Number of training epochs.
        warmup_steps (int): Number of warmup steps.
        per_device_train_batch_size (int): Batch size per device.
        weight_decay (float): Weight decay.
        logging_steps (int): Number of steps between logging.
        evaluation_strategy (str): Strategy for evaluation.
        eval_steps (int): Number of steps between evaluations.
        save_steps (int): Number of steps between model saves.
        gradient_accumulation_steps (int): Number of gradient accumulation steps.
    """
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: int
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Configuration for model evaluation.

    Attributes:
        root_dir (Path): Directory for storing evaluation artifacts.
        data_path (Path): Path to the evaluation data.
        model_path (Path): Path to the trained model.
        tokenizer_path (Path): Path to the tokenizer.
        metric_file_name (Path): Name of the file to store evaluation metrics.
    """
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: str
