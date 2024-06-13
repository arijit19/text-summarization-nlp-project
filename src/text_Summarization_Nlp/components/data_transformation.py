import os
from pathlib import Path
from text_Summarization_Nlp.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from text_Summarization_Nlp.entity import DataTransformationConfig
from typing import Dict, Any

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """
        Initialize the DataTransformation class with the provided configuration.
        
        Args:
            config (DataTransformationConfig): Configuration for data transformation.
        """
        try:
            self.config = config
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            logger.info(f"Tokenizer loaded from {config.tokenizer_name}")
        except Exception as e:
            logger.error(f"Failed to initialize DataTransformation: {e}")
            raise e

    def convert_examples_to_features(self, example_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert examples to features for model training.

        Args:
            example_batch (Dict[str, Any]): A batch of examples.

        Returns:
            Dict[str, Any]: Encoded inputs and targets.
        """
        try:
            input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)

            return {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': target_encodings['input_ids']
            }
        except Exception as e:
            logger.error(f"Error in converting examples to features: {e}")
            raise e

    def convert(self):
        """
        Perform the data transformation by converting and saving the dataset.
        """
        try:
            # Convert Path object to string before using it with load_from_disk
            data_path_str = str(self.config.data_path)
            dataset_samsum = load_from_disk(data_path_str)
            logger.info(f"Dataset loaded from {data_path_str}")

            dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
            logger.info("Dataset transformation complete")

            output_path = Path(self.config.root_dir) / "samsum_dataset"
            dataset_samsum_pt.save_to_disk(str(output_path))
            logger.info(f"Transformed dataset saved to {output_path}")
        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise e
