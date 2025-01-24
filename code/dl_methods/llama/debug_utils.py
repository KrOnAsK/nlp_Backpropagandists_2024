import torch
import pandas as pd
from data_utils import get_narrative_key

def debug_misclassifications(dataset, model, tokenizer, label_mapping, dataset_type="Training"):
    """Debug misclassified examples with detailed output and proper device handling"""
    try:
        print(f"\nAnalyzing misclassifications in {dataset_type} dataset...")
        
        # Determine device and ensure model is on it
        device = next(model.parameters()).device
        print(f"Model is on device: {device}")
        
        # Prepare data
        texts = dataset['tokens_normalized'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else x
        ).tolist()
        
        true_labels = torch.tensor([
            label_mapping[get_narrative_key(eval(n)[0] if isinstance(n, str) else n[0])]
            for n in dataset['narrative_subnarrative_pairs']
        ]).to(device)
        
        print(f"\nTotal samples to analyze: {len(texts)}")

        # Get predictions in batches to manage memory
        batch_size = 8
        predictions = []
        confidences = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = tokenizer(
            batch_texts, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
            )
            # Explicitly move each tensor to the device
            encodings = {k: v.to(device) for k, v in encodings.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**encodings)  # Use unpacking to handle all inputs
                batch_preds = outputs.logits.argmax(-1)
                batch_confs = torch.softmax(outputs.logits, dim=-1).max(dim=-1)[0]
                
                # Collect predictions and confidences (keep on GPU for now)
                predictions.append(batch_preds)
                confidences.append(batch_confs)

        # Concatenate all batches
        predictions = torch.cat(predictions).cpu().numpy()  # Move to CPU only at the end
        confidences = torch.cat(confidences).cpu().numpy()
        true_labels = true_labels.cpu().numpy()

        # Track misclassifications
        misclassifications = []
        for idx, (pred, true, conf) in enumerate(zip(predictions, true_labels, confidences)):
            if pred != true:
                misclassifications.append({
                    'text': texts[idx][:200],  # First 200 chars for brevity
                    'predicted': pred,
                    'actual': true,
                    'confidence': conf,
                    'dataset_type': dataset_type
                })

        # Create DataFrame and display results
        misclass_df = pd.DataFrame(misclassifications)
        
        print(f"\nTotal misclassifications: {len(misclass_df)}")
        print(f"Accuracy: {1 - len(misclass_df)/len(texts):.4f}")
        
        if len(misclass_df) > 0:
            print("\nMisclassification distribution:")
            print(misclass_df.groupby(['actual', 'predicted']).size().unstack(fill_value=0))
            
            print("\nSample misclassifications:")
            for i, row in misclass_df.head().iterrows():
                print(f"\nExample {i+1}:")
                print(f"Text: {row['text']}")
                print(f"Predicted: {row['predicted']}, Actual: {row['actual']}")
                print(f"Confidence: {row['confidence']:.4f}")
        
        return misclass_df

    except Exception as e:
        print(f"Error in debugging misclassifications: {str(e)}")
        import traceback
        traceback.print_exc()
        raise