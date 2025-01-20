from transformers import TrainingArguments, Trainer
import wandb
import os
from datetime import datetime
import torch
import logging
import traceback

from model import compute_metrics

def train_model(model, train_dataset, val_dataset, output_dir, current_date):
    """Train the model in two phases: head pre-training and full model fine-tuning"""
    try:
        # Pre-train classification head
        print("\nStarting classification head pre-training...")
        
        # Freeze LoRA adapters
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = False

        # Training arguments for head pre-training
        head_training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "head_pretraining"),
            run_name=f"llama-head-pretraining-{current_date}",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=1e-3,
            warmup_ratio=0.1,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_dir=os.path.join(output_dir, "head_logs"),
            logging_steps=10,
            remove_unused_columns=False,
            report_to="wandb"
        )
        
        # Initialize trainer for head pre-training
        head_trainer = Trainer(
            model=model,
            args=head_training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train classification head
        head_trainer.train()

        # Unfreeze LoRA adapters for full training
        print("\nUnfreezing LoRA adapters for full training...")
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

        # Training arguments for full model fine-tuning
        print("\nStarting full model fine-tuning...")
        log_dir = os.path.join(output_dir, "logs")
        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=f"llama-classification-run-{current_date}",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=log_dir,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            logging_steps=10,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            remove_unused_columns=False,
            report_to="wandb"
        )

        # Initialize trainer for full model fine-tuning
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        # Train full model
        trainer.train()

        return trainer

    except Exception as e:
        print(f"Error in training: {str(e)}")
        traceback.print_exc()
        raise
