from sklearn.model_selection import train_test_split
from transformers import LlamaTokenizer
from datasets import Dataset
import pandas as pd
import json
import os
import torch

def get_narrative_key(narrative_dict):
    """Extract key from narrative dictionary for classification"""
    if isinstance(narrative_dict, str):
        narrative_dict = eval(narrative_dict)
    return narrative_dict['narrative']

def prepare_data(df, model_name, output_dir):
    """Prepare data for training"""
    # Create narrative mapping
    print("\nCreating narrative mapping...")
    narratives = df['narrative_subnarrative_pairs'].apply(
        lambda x: eval(x)[0] if isinstance(x, str) else x[0]
    ).tolist()
    
    unique_narratives = set(get_narrative_key(n) for n in narratives)
    label_mapping = {narrative: idx for idx, narrative in enumerate(sorted(unique_narratives))}
    
    print(f"Number of unique narratives: {len(unique_narratives)}")
    print("\nSample narrative mappings:")
    for i, (narrative, idx) in enumerate(list(label_mapping.items())[:5]):
        print(f"{idx}: {narrative}")

    # Save label mapping
    with open(os.path.join(output_dir, "label_mapping.json"), 'w') as f:
        json.dump(label_mapping, f, indent=2)

    # Split data
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")

    # Process texts and labels
    train_texts = df_train['tokens_normalized'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else x
    ).tolist()
    val_texts = df_val['tokens_normalized'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else x
    ).tolist()
    
    train_labels = [
        label_mapping[get_narrative_key(eval(n)[0] if isinstance(n, str) else n[0])]
        for n in df_train['narrative_subnarrative_pairs']
    ]
    val_labels = [
        label_mapping[get_narrative_key(eval(n)[0] if isinstance(n, str) else n[0])]
        for n in df_val['narrative_subnarrative_pairs']
    ]

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize texts - Changed to not return tensors
    print("\nTokenizing texts...")
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding=True, 
        max_length=512,
        return_tensors=None  # Changed this line
    )
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        padding=True, 
        max_length=512,
        return_tensors=None  # Changed this line
    )

    # Create datasets without converting to tensors yet
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })

    # Set the format to PyTorch but don't specify device
    train_dataset.set_format('torch')
    val_dataset.set_format('torch')

    return train_dataset, val_dataset, tokenizer, label_mapping, len(label_mapping)

# utils.py
import torch
import pandas as pd 

def ensure_model_on_device(model):
    """Ensure model is on the correct device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

def prepare_data_for_model(batch_texts, tokenizer, device):
    """Prepare data for model input"""
    encodings = tokenizer(
        batch_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move entire encoding dict to device
    encodings = {k: v.to(device) for k, v in encodings.items()}
    return encodings

def get_predictions_batch(model, batch_texts, tokenizer, device):
    """Get predictions for a batch of texts"""
    model.eval()  # Ensure model is in eval mode
    encodings = prepare_data_for_model(batch_texts, tokenizer, device)
    
    with torch.no_grad():
        outputs = model(**encodings)
        batch_preds = outputs.logits.argmax(-1)
        batch_confs = torch.softmax(outputs.logits, dim=-1).max(dim=-1)[0]
    
    return batch_preds, batch_confs