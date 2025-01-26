import torch
from transformers import (
    LlamaTokenizer,
    LlamaForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import wandb
import os
from datetime import datetime
import json
from datasets import Dataset
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import numpy as np

from data_utils import get_narrative_key


def initialize_model(model_name, num_labels):
    """Initialize the LLaMA model with quantization configuration"""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = LlamaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )

    return model


def setup_peft(model):
    """Configure and apply LoRA adapters"""
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model
