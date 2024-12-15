import torch
import wandb
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datetime import datetime
import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

class CustomDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading BERT input data"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def create_label_mapping(all_narratives):
    """
    Create a consistent mapping for all narrative pairs
    
    Args:
        all_narratives: List of lists of narrative dictionaries
    
    Returns:
        dict: Mapping from narrative string to numeric index
    """
    unique_narratives = set()
    for narratives in all_narratives:
        for narrative in narratives:
            narrative_str = str(narrative)  # Convert dict to string
            unique_narratives.add(narrative_str)
    
    # Create mapping
    narrative_to_idx = {
        narrative: idx 
        for idx, narrative in enumerate(sorted(unique_narratives))
    }
    
    logger.info(f"Created mapping for {len(narrative_to_idx)} unique narratives")
    return narrative_to_idx

def get_first_narrative_label(narrative_list, label_mapping):
    """
    Convert first narrative in list to numeric label
    
    Args:
        narrative_list: List of narrative dictionaries
        label_mapping: Dictionary mapping narrative strings to indices
    
    Returns:
        int: Numeric label for the first narrative
    """
    if narrative_list and len(narrative_list) > 0:
        narrative_str = str(narrative_list[0])
        return label_mapping[narrative_str]
    return None

def prepare_data(df, label_mapping=None):
    """
    Prepare data for BERT training
    
    Args:
        df: DataFrame containing tokens_normalized and narrative_subnarrative_pairs
        label_mapping: Optional pre-existing label mapping to use
    
    Returns:
        tuple: (texts, labels, label_mapping)
    """
    try:
        # Handle tokens_normalized
        texts = df['tokens_normalized'].tolist()
        texts = [' '.join(tokens) if isinstance(tokens, list) else tokens for tokens in texts]
        
        # Convert narrative_subnarrative_pairs to list if it's a string
        narratives = df['narrative_subnarrative_pairs'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        ).tolist()

        # Create or use label mapping
        if label_mapping is None:
            label_mapping = create_label_mapping(narratives)
            
        # Convert narratives to numerical labels
        labels = []
        for narrative_list in narratives:
            if narrative_list:  # Check if list is not empty
                label_str = str(narrative_list[0])  # Convert first narrative dict to string
                if label_str in label_mapping:
                    labels.append(label_mapping[label_str])
                else:
                    raise ValueError(f"Unknown narrative: {label_str}")
            else:
                raise ValueError("Empty narrative list found")

        logger.info(f"Number of unique labels in mapping: {len(label_mapping)}")
        logger.info(f"Sample text: {texts[0][:100]}")
        logger.info(f"Sample label: {labels[0]}")
        
        return texts, labels, label_mapping

    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        logger.error(f"Sample narrative_subnarrative_pairs: {df['narrative_subnarrative_pairs'].iloc[0]}")
        raise

def compute_metrics(pred):
    """
    Compute evaluation metrics
    
    Args:
        pred: Prediction object from trainer
    
    Returns:
        dict: Dictionary containing computed metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)

    # Receiving all unique classes
    unique_classes = np.unique(labels)
    cm_per_class = {}

    # Creating Confusion Matrix for each Class
    for class_idx in unique_classes:
        binary_labels = (labels == class_idx).astype(int)
        binary_preds = (preds == class_idx).astype(int)

        cm = confusion_matrix(binary_labels, binary_preds)
        cm_per_class[f"Class_{class_idx}"] = cm.tolist()

        print(f"\nConfusion Matrix for Class {class_idx}:")
        print(cm)


    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm_per_class
    }

def train_bert(df, base_path, project_name="bert-finetuning", min_examples_per_class=2):
    """
    Train BERT model on the provided DataFrame
    
    Args:
        df: DataFrame containing the training data
        base_path: Base path for saving model outputs
        project_name: Name for the wandb project
        min_examples_per_class: Minimum number of examples required for each class
    
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

        # Get all narratives and count their frequencies
        all_narratives = df['narrative_subnarrative_pairs'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        ).tolist()
        
        # Count frequencies of each narrative
        narrative_counts = {}
        for narratives in all_narratives:
            if narratives:
                narrative_str = str(narratives[0])  # Use first narrative
                narrative_counts[narrative_str] = narrative_counts.get(narrative_str, 0) + 1
        
        # Filter narratives that have enough examples
        valid_narratives = {
            narrative: count 
            for narrative, count in narrative_counts.items() 
            if count >= min_examples_per_class
        }
        
        logger.info(f"Total unique narratives: {len(narrative_counts)}")
        logger.info(f"Narratives with >= {min_examples_per_class} examples: {len(valid_narratives)}")
        
        # Create mapping only for valid narratives
        label_mapping = {
            narrative: idx 
            for idx, narrative in enumerate(sorted(valid_narratives.keys()))
        }
        
        # Filter DataFrame to only include rows with valid narratives
        df['temp_narrative'] = df['narrative_subnarrative_pairs'].apply(
            lambda x: str(eval(x)[0] if isinstance(x, str) else x[0])
        )
        df_filtered = df[df['temp_narrative'].isin(valid_narratives.keys())].copy()
        df_filtered.drop('temp_narrative', axis=1, inplace=True)
        
        logger.info(f"Original dataset size: {len(df)}")
        logger.info(f"Filtered dataset size: {len(df_filtered)}")
        
        # Save label mapping
        mapping_path = os.path.join(output_dir, "label_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        # Convert narratives to numeric labels for stratification
        stratify_labels = df_filtered['narrative_subnarrative_pairs'].apply(
            lambda x: get_first_narrative_label(eval(x) if isinstance(x, str) else x, label_mapping)
        )
        
        # Split data using numeric labels for stratification
        df_train, df_val = train_test_split(
            df_filtered, 
            test_size=0.2, 
            random_state=42, 
            stratify=stratify_labels
        )
        
        logger.info(f"Training set size: {len(df_train)}, Validation set size: {len(df_val)}")

        # Prepare data using the complete mapping
        train_texts, train_labels, _ = prepare_data(df_train, label_mapping)
        val_texts, val_labels, _ = prepare_data(df_val, label_mapping)

        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Tokenize texts
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        # Create datasets
        train_dataset = CustomDataset(train_encodings, train_labels)
        val_dataset = CustomDataset(val_encodings, val_labels)

        # Initialize model with correct number of labels
        num_labels = len(label_mapping)
        logger.info(f"Number of unique labels: {num_labels}")
        
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels=num_labels
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=log_dir,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            logging_steps=10,
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

        # Save model and tokenizer
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model and tokenizer saved to {output_dir}")

        # End wandb run
        wandb.finish()

        return results

    except Exception as e:
        logger.error(f"Error in BERT training: {str(e)}")
        wandb.finish()
        raise

def predict(text, model_path):
    """
    Make predictions using a trained model
    
    Args:
        text: Input text to classify
        model_path: Path to the saved model
    
    Returns:
        int: Predicted class index
    """
    try:
        # Load model and tokenizer
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # Load label mapping
        with open(os.path.join(model_path, "label_mapping.json"), 'r') as f:
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


def debug_misclassifications(dataset, dataset_type="Training", min_examples_per_class=2):
    """
    Function to debug misclassified narratives, showing tokens, predictions,
    actual labels, and the dataset type.

    Args:
        dataset (DataFrame): The dataset (either training or test) to analyze
        dataset_type (str): The type of dataset ('Training' or 'Testing')
        min_examples_per_class (int): Minimum number of examples required for each class

    Returns:
        misclassification_df (DataFrame): DataFrame with misclassified narratives and their details
    """
    try:
        current_date = datetime.now().strftime("%Y%m%d")

        # Get all narratives and count their frequencies
        all_narratives = dataset['narrative_subnarrative_pairs'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        ).tolist()

        # Count frequencies of each narrative
        narrative_counts = {}
        for narratives in all_narratives:
            if narratives:
                narrative_str = str(narratives[0])  # Use first narrative
                narrative_counts[narrative_str] = narrative_counts.get(narrative_str, 0) + 1

        # Filter narratives that have enough examples
        valid_narratives = {
            narrative: count
            for narrative, count in narrative_counts.items()
            if count >= min_examples_per_class
        }

        # Create label mapping only for valid narratives
        label_mapping = {
            narrative: idx
            for idx, narrative in enumerate(sorted(valid_narratives.keys()))
        }

        # Ensure the model is defined inside the function
        # Load the pre-trained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_mapping))

        # Prepare the dataset for predictions
        texts = dataset['narrative_subnarrative_pairs'].apply(
            lambda x: eval(x)[0] if isinstance(x, str) else x[0]).tolist()
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

        # Create a dataset for inference
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])

        # Predict using the model
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        # Create a DataFrame to track misclassifications
        misclassifications = []
        for idx, row in dataset.iterrows():
            # Get the actual label
            actual_label = row['narrative_subnarrative_pairs']
            actual_label_str = str(eval(actual_label)[0])  # Extract the narrative as a string
            actual_label_idx = label_mapping.get(actual_label_str, -1)

            # Get the predicted label
            predicted_label = predictions[idx].item()

            if actual_label_idx != predicted_label:
                # If the prediction is incorrect, track the details
                misclassified_narrative = eval(row['narrative_subnarrative_pairs'])[0]  # Get the narrative
                misclassifications.append({
                    'narrative': misclassified_narrative,
                    'predicted_label': predicted_label,
                    'actual_label': actual_label_idx,
                    'dataset_type': dataset_type
                })

        # Create a DataFrame from the misclassified narratives
        misclassification_df = pd.DataFrame(misclassifications)

        # Log misclassifications (can also be logged in wandb if needed)
        print(f"Misclassified entries in {dataset_type}:")
        print(misclassification_df)

        return misclassification_df

    except Exception as e:
        print(f"Error in debugging misclassifications: {str(e)}")
