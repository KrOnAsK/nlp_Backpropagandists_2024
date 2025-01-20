# modules/utils.py

import logging
import sys
import pandas as pd
import torch
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

def setup_logging():
    """
    Setup logging configuration
    """
    # Create logs directory if it doesn't exist
    log_filename = f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )


def compute_metrics(pred):
    """
    Compute evaluation metrics
    
    Args:
        pred: Prediction object from trainer, containing label_ids and predictions
    
    Returns:
        dict: Dictionary containing computed metrics:
            - accuracy: Overall accuracy score
            - f1: Weighted F1 score
            - precision: Weighted precision score
            - recall: Weighted recall score
            - confusion_matrix: Per-class confusion matrices
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted",zero_division=0)
    acc = accuracy_score(labels, preds)

    # Computing confusion matrix and metrics per class
    unique_classes = np.unique(labels)
    cm_per_class = {}
    
    for class_idx in unique_classes:
        binary_labels = (labels == class_idx).astype(int)
        binary_preds = (preds == class_idx).astype(int)
        
        # Compute and store confusion matrix
        cm = confusion_matrix(binary_labels, binary_preds)
        cm_per_class[f"Class_{class_idx}"] = cm.tolist()
        
        # Print per-class metrics
        print(f"\nMetrics for Class {class_idx}:")
        print(f"Confusion Matrix:\n{cm}")
        class_precision = precision_recall_fscore_support(binary_labels, binary_preds, average='binary')[0]
        class_recall = precision_recall_fscore_support(binary_labels, binary_preds, average='binary')[1]
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        print(f"Precision: {class_precision:.4f}")
        print(f"Recall: {class_recall:.4f}")
        print(f"F1 Score: {class_f1:.4f}")

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm_per_class
    }

logger = logging.getLogger(__name__)

def debug_misclassifications(dataset, model, tokenizer, label_mapping, 
                           dataset_type: str = "Training") -> pd.DataFrame:
    """
    Debug misclassified narratives
    
    Args:
        dataset: DataFrame containing the narratives
        model: trained model
        tokenizer: tokenizer for the model
        label_mapping: dictionary mapping labels to indices
        dataset_type: string indicating the type of dataset (default: "Training")
        
    Returns:
        pd.DataFrame: DataFrame containing misclassified examples
    """
    try:
        # Safely extract and process the narrative texts
        def extract_narrative(x):
            try:
                if isinstance(x, str):
                    # If it's a string, evaluate it and get the first element
                    return eval(x)[0]
                elif isinstance(x, (list, tuple)):
                    # If it's already a list/tuple, get the first element
                    return x[0]
                else:
                    logger.warning(f"Unexpected data type for narrative: {type(x)}")
                    return str(x)  # Convert to string as fallback
            except Exception as e:
                logger.warning(f"Error processing narrative: {str(e)}")
                return str(x)  # Convert to string as fallback

        # Process texts and ensure they're strings
        texts = dataset['narrative_subnarrative_pairs'].apply(extract_narrative).tolist()
        texts = [str(t) for t in texts]  # Ensure all items are strings
        
        # Tokenize the processed texts
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

        # Convert to tensors
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])

        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Process misclassifications
        misclassifications = []
        for idx, row in dataset.iterrows():
            try:
                # Safely process actual label
                actual_label = row['narrative_subnarrative_pairs']
                if isinstance(actual_label, str):
                    actual_label_str = str(eval(actual_label)[0])
                else:
                    actual_label_str = str(actual_label[0])
                
                actual_label_idx = label_mapping.get(actual_label_str, -1)
                predicted_label = predictions[idx].item()

                if actual_label_idx != predicted_label:
                    misclassified_narrative = texts[idx]  # Use the processed text
                    misclassifications.append({
                        'narrative': misclassified_narrative,
                        'predicted_label': predicted_label,
                        'actual_label': actual_label_idx,
                        'dataset_type': dataset_type
                    })
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {str(e)}")
                continue

        return pd.DataFrame(misclassifications)

    except Exception as e:
        logger.error(f"Error in debugging misclassifications: {str(e)}")
        raise