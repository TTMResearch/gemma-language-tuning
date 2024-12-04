import torch
from transformers import Trainer, TrainingArguments
from typing import Dict, Optional
import wandb
from loguru import logger

class AfrikaansTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset=None,
        training_args: Optional[Dict] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        # Default training arguments
        default_args = {
            "output_dir": "outputs",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "logging_steps": 10,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "report_to": "wandb"
        }
        
        # Update with custom args if provided
        if training_args:
            default_args.update(training_args)
            
        self.training_args = TrainingArguments(**default_args)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )
        
    def train(self):
        """Start training"""
        logger.info("Starting training...")
        try:
            # Initialize wandb
            wandb.init(project="gemma-afrikaans", name="afrikaans-finetune")
            
            # Train the model
            train_result = self.trainer.train()
            
            # Log metrics
            metrics = train_result.metrics
            logger.info(f"Training metrics: {metrics}")
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise e
        
    def save_model(self, output_dir: str):
        """Save the trained model"""
        self.trainer.save_model(output_dir)
        logger.info(f"Model saved to {output_dir}")