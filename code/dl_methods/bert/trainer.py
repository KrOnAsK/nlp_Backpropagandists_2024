import os
import sys
code_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(code_path)

import json
import logging
import wandb
from datetime import datetime
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TrainingArguments, Trainer

from dataset import prepare_data, CustomDataset, get_first_narrative_label
from model import initialize_model
from modules.utils import compute_metrics

logger = logging.getLogger(__name__)

def train_bert(df, base_path: str, project_name: str = "bert-finetuning", 
               min_examples_per_class: int = 2) -> Dict[str, float]:
    """
    Train BERT model on the provided DataFrame
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

        # Get narrative counts and filter valid narratives
        all_narratives = df['narrative_subnarrative_pairs'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        ).tolist()
        
        narrative_counts = {}
        for narratives in all_narratives:
            if narratives:
                narrative_str = str(narratives[0])
                narrative_counts[narrative_str] = narrative_counts.get(narrative_str, 0) + 1
        
        valid_narratives = {
            narrative: count 
            for narrative, count in narrative_counts.items() 
            if count >= min_examples_per_class
        }
        
        # Create label mapping and filter data
        label_mapping = {
            narrative: idx 
            for idx, narrative in enumerate(sorted(valid_narratives.keys()))
        }
        
        df['temp_narrative'] = df['narrative_subnarrative_pairs'].apply(
            lambda x: str(eval(x)[0] if isinstance(x, str) else x[0])
        )
        df_filtered = df[df['temp_narrative'].isin(valid_narratives.keys())].copy()
        df_filtered.drop('temp_narrative', axis=1, inplace=True)
        
        # Save label mapping
        with open(os.path.join(output_dir, "label_mapping.json"), 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        # Split data
        stratify_labels = df_filtered['narrative_subnarrative_pairs'].apply(
            lambda x: get_first_narrative_label(eval(x) if isinstance(x, str) else x, label_mapping)
        )
        
        df_train, df_val = train_test_split(
            df_filtered, 
            test_size=0.2, 
            random_state=42, 
            stratify=stratify_labels
        )

        # Prepare data
        train_texts, train_labels, _ = prepare_data(df_train, label_mapping)
        val_texts, val_labels, _ = prepare_data(df_val, label_mapping)

        # Initialize tokenizer and encode data
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        # Create datasets
        train_dataset = CustomDataset(train_encodings, train_labels)
        val_dataset = CustomDataset(val_encodings, val_labels)

        # Initialize model
        model = initialize_model(len(label_mapping))

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=f"bert-base-uncased-custom-{current_date}",
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

        # Initialize and run trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        results = trainer.evaluate()
        
        # Save model and tokenizer
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        wandb.finish()
        return results

    except Exception as e:
        logger.error(f"Error in BERT training: {str(e)}")
        wandb.finish()
        raise