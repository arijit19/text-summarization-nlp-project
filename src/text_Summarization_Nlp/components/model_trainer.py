# import os
# import torch
# from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
# from datasets import load_from_disk
# from text_Summarization_Nlp.entity import ModelTrainerConfig
# from text_Summarization_Nlp.logging import logger

# class ModelTrainer:
#     def __init__(self, config: ModelTrainerConfig):
#         """
#         Initialize the ModelTrainer with the provided configuration.
        
#         Args:
#             config (ModelTrainerConfig): Configuration for model training.
#         """
#         self.config = config

#     def train(self):
#         """
#         Train the model using the specified configuration.
#         """
#         try:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             logger.info(f"Using device: {device}")
            
#             tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
#             model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
#             logger.info(f"Model and tokenizer loaded from {self.config.model_ckpt}")
            
#             seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
            
#             # Load dataset
#             dataset_samsum_pt = load_from_disk(self.config.data_path)
#             logger.info(f"Dataset loaded from {self.config.data_path}")

#             trainer_args = TrainingArguments(
#                 output_dir=self.config.root_dir,
#                 num_train_epochs=self.config.num_train_epochs,
#                 warmup_steps=self.config.warmup_steps,
#                 per_device_train_batch_size=self.config.per_device_train_batch_size,
#                 per_device_eval_batch_size=self.config.per_device_train_batch_size,
#                 weight_decay=self.config.weight_decay,
#                 logging_steps=self.config.logging_steps,
#                 evaluation_strategy=self.config.evaluation_strategy,
#                 eval_steps=self.config.eval_steps,
#                 save_steps=self.config.save_steps,
#                 gradient_accumulation_steps=self.config.gradient_accumulation_steps
#             )

#             trainer = Trainer(
#                 model=model_pegasus,
#                 args=trainer_args,
#                 tokenizer=tokenizer,
#                 data_collator=seq2seq_data_collator,
#                 train_dataset=dataset_samsum_pt["train"],
#                 eval_dataset=dataset_samsum_pt["validation"]
#             )
            
#             logger.info("Starting model training")
#             trainer.train()
#             logger.info("Model training completed")
            
#             # Save model and tokenizer
#             model_save_path = os.path.join(self.config.root_dir, "pegasus-samsum-model")
#             tokenizer_save_path = os.path.join(self.config.root_dir, "tokenizer")
#             model_pegasus.save_pretrained(model_save_path)
#             tokenizer.save_pretrained(tokenizer_save_path)
#             logger.info(f"Model saved to {model_save_path}")
#             logger.info(f"Tokenizer saved to {tokenizer_save_path}")

#         except Exception as e:
#             logger.error(f"An error occurred during training: {e}")
#             raise e


import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from text_Summarization_Nlp.entity import ModelTrainerConfig
from text_Summarization_Nlp.logging import logger

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize the ModelTrainer with the provided configuration.
        
        Args:
            config (ModelTrainerConfig): Configuration for model training.
        """
        self.config = config

    def train(self):
        """
        Train the model using the specified configuration.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Ensure model_ckpt uses forward slashes
            model_ckpt = str(self.config.model_ckpt).replace("\\", "/")
            tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
            logger.info(f"Model and tokenizer loaded from {model_ckpt}")
            
            seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
            
            # Ensure data_path uses forward slashes
            data_path_str = str(self.config.data_path).replace("\\", "/")
            dataset_samsum_pt = load_from_disk(data_path_str)
            logger.info(f"Dataset loaded from {data_path_str}")

            # Ensure root_dir uses forward slashes
            root_dir_str = str(self.config.root_dir).replace("\\", "/")
            trainer_args = TrainingArguments(
                output_dir=root_dir_str,
                num_train_epochs=self.config.num_train_epochs,
                warmup_steps=self.config.warmup_steps,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_train_batch_size,
                weight_decay=self.config.weight_decay,
                logging_steps=self.config.logging_steps,
                evaluation_strategy=self.config.evaluation_strategy,
                eval_steps=self.config.eval_steps,
                save_steps=self.config.save_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps
            )

            trainer = Trainer(
                model=model_pegasus,
                args=trainer_args,
                tokenizer=tokenizer,
                data_collator=seq2seq_data_collator,
                train_dataset=dataset_samsum_pt["train"],
                eval_dataset=dataset_samsum_pt["validation"]
            )
            
            logger.info("Starting model training")
            trainer.train()
            logger.info("Model training completed")
            
            # Save model and tokenizer
            model_save_path = os.path.join(root_dir_str, "pegasus-samsum-model")
            tokenizer_save_path = os.path.join(root_dir_str, "tokenizer")
            model_pegasus.save_pretrained(model_save_path)
            tokenizer.save_pretrained(tokenizer_save_path)
            logger.info(f"Model saved to {model_save_path}")
            logger.info(f"Tokenizer saved to {tokenizer_save_path}")

        except Exception as e:
            logger.error(f"An error occurred during training: {e}")
            raise e
