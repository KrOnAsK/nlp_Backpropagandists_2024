import torch
import json
import logging
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Tuple, List, Dict
import os

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_path: str,
) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    """Load pre-trained model and tokenizer"""
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer


def predict(text: str, model_path: str) -> Tuple[str, List[float]]:
    """
    Make predictions using a trained model
    """
    try:
        model, tokenizer = load_model_and_tokenizer(model_path)

        # Load label mapping
        with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
            label_mapping = json.load(f)

        # Prepare input
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")

        # Get prediction
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()

        # Convert to original label
        idx_to_label = {v: k for k, v in label_mapping.items()}
        predicted_label = idx_to_label[predicted_class]

        return predicted_label, outputs.logits.softmax(-1)[0].tolist()

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise


def initialize_model(num_labels: int) -> BertForSequenceClassification:
    """Initialize BERT model with specified number of labels"""
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    )
