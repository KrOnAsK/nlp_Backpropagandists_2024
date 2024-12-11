#!/usr/bin/env python3
"""
BERT-based classification model for CoNLL-U formatted data.
"""

import os
import logging
import conllu
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConlluDataset(torch.utils.data.Dataset):
    """Custom Dataset for handling CoNLL-U data."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def load_conllu_data(filepath):
    """
    Load and process CoNLL-U file into a pandas DataFrame.
    
    Args:
        filepath (str): Path to CoNLL-U file
    
    Returns:
        pandas.DataFrame: DataFrame with text and label columns
    """
    logger.info(f"Loading data from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = conllu.parse(f.read())
        
        texts = []
        labels = []
        
        for sentence in data:
            # Join all tokens to form the complete text
            text = ' '.join([token['form'] for token in sentence])
            # Modify this based on where your labels are stored in the CoNLL-U file
            label = sentence.metadata.get('label', 0)  
            
            texts.append(text)
            labels.append(int(label))  # Ensure labels are integers
        
        return pd.DataFrame({'text': texts, 'label': labels})
    
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise

def compute_metrics(pred):
    """
    Compute evaluation metrics.
    
    Args:
        pred: Prediction object from trainer
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro"
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load data
    try:
        tr_data = load_conllu_data('train.conllu')
        val_data = load_conllu_data('test.conllu')
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return

    # Convert to lists and ensure proper formatting
    train_texts = tr_data.text.tolist()
    train_labels = [int(label) for label in tr_data.label.tolist()]
    val_texts = val_data.text.tolist()
    val_labels = [int(label) for label in val_data.label.tolist()]

    # Debug information
    logger.info(f"Number of unique labels: {len(set(train_labels))}")
    logger.info(f"Sample labels: {train_labels[:5]}")
    logger.info(f"Label type: {type(train_labels[0])}")

    # Initialize tokenizer and encode texts
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=512
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=512
    )

    # Create datasets
    train_dataset = ConlluDataset(train_encodings, train_labels)
    val_dataset = ConlluDataset(val_encodings, val_labels)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    # Initialize model
    num_labels = len(set(train_labels))
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train model
    try:
        trainer.train()
        
        # Save the model
        output_dir = "./model"
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()