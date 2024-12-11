import torch
import wandb
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def prepare_data(df):
    """Prepare data for BERT training"""
    texts = df['content'].tolist()
    labels = df['target_indices'].apply(eval).apply(lambda x: x[0]).tolist()
    return texts, labels

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_bert(df, base_path, project_name="bert-finetuning"):
    """
    Train BERT model on the provided DataFrame
    
    Args:
        df: DataFrame containing the preprocessed data
        base_path: Base path for saving model and logs
        project_name: Name for the wandb project
    
    Returns:
        dict: Evaluation results
    """
    try:
        current_date = datetime.now().strftime("%Y%m%d")
        
        # Create output directories
        output_dir = os.path.join(base_path, f"models/bert_{current_date}")
        log_dir = os.path.join(base_path, f"logs/bert_{current_date}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize wandb
        wandb.init(project=project_name, name=f"bert-base-uncased-custom-{current_date}")

        # Split data
        df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
        logger.info(f"Training set size: {len(df_train)}, Validation set size: {len(df_val)}")

        # Prepare data
        train_texts, train_labels = prepare_data(df_train)
        val_texts, val_labels = prepare_data(df_val)

        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Tokenize texts
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)

        # Create datasets
        train_dataset = CustomDataset(train_encodings, train_labels)
        val_dataset = CustomDataset(val_encodings, val_labels)

        # Calculate number of labels
        num_labels = len(set(df['target_indices'].apply(eval).apply(lambda x: x[0])))
        logger.info(f"Number of labels: {num_labels}")

        # Initialize model
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=log_dir,
            load_best_model_at_end=True,
            greater_is_better=False,
            logging_steps=10,
            metric_for_best_model='eval_loss',
            report_to="wandb"
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        # Train model
        logger.info("Starting training...")
        trainer.train()

        # Evaluate model
        logger.info("Evaluating model...")
        results = trainer.evaluate()
        
        # Log results
        wandb.log(results)
        logger.info(f"Evaluation results: {results}")

        # Save model
        model_save_path = os.path.join(output_dir, "final_model")
        trainer.save_model(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        # End wandb run
        wandb.finish()

        return results

    except Exception as e:
        logger.error(f"Error in BERT training: {str(e)}")
        raise