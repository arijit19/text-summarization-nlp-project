import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, load_metric
from text_Summarization_Nlp.entity import ModelEvaluationConfig
from text_Summarization_Nlp.logging import logger
from typing import List, Generator, Dict

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize the ModelEvaluation with the provided configuration.
        
        Args:
            config (ModelEvaluationConfig): Configuration for model evaluation.
        """
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements: List[str], batch_size: int) -> Generator[List[str], None, None]:
        """
        Split the dataset into smaller batches that can be processed simultaneously.
        
        Args:
            list_of_elements (List[str]): List of elements to split.
            batch_size (int): Size of each batch.

        Yields:
            List[List[str]]: Batches of elements.
        """
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer,
                                    batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu",
                                    column_text="article",
                                    column_summary="highlights") -> Dict[str, float]:
        """
        Calculate the metric on the test dataset.

        Args:
            dataset: The dataset to evaluate.
            metric: The metric to calculate.
            model: The model to use for evaluation.
            tokenizer: The tokenizer to use.
            batch_size (int): Batch size.
            device (str): Device to use (cuda or cpu).
            column_text (str): Column containing the text.
            column_summary (str): Column containing the summaries.

        Returns:
            Dict[str, float]: The calculated metric scores.
        """
        try:
            article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
            target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

            for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
                inputs = tokenizer(article_batch, max_length=1024, truncation=True,
                                   padding="max_length", return_tensors="pt")
                
                summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                           attention_mask=inputs["attention_mask"].to(device),
                                           length_penalty=0.8, num_beams=8, max_length=128)

                decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)
                                     for s in summaries]
                decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
                metric.add_batch(predictions=decoded_summaries, references=target_batch)

            score = metric.compute()
            return score
        except Exception as e:
            logger.error(f"Error calculating metric on test dataset: {e}")
            raise e

    def evaluate(self):
        """
        Perform model evaluation.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
            logger.info("Model and tokenizer loaded successfully.")
            
            dataset_samsum_pt = load_from_disk(self.config.data_path)
            logger.info(f"Dataset loaded from {self.config.data_path}")

            rouge_metric = load_metric('rouge')
            score = self.calculate_metric_on_test_ds(
                dataset_samsum_pt['test'], rouge_metric, model_pegasus, tokenizer, 
                batch_size=2, column_text='dialogue', column_summary='summary'
            )

            rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
            rouge_dict = {rn: score[rn].mid.fmeasure for rn in rouge_names}

            df = pd.DataFrame(rouge_dict, index=['pegasus'])
            df.to_csv(self.config.metric_file_name, index=False)
            logger.info(f"Metrics saved to {self.config.metric_file_name}")
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise e