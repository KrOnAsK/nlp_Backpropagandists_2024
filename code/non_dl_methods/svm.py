from modules.modules_svm.preprocessing_svm import create_label_mapping
from modules.modules_svm.preprocessing_svm import prepare_data
from modules.modules_svm.preprocessing_svm import bag_of_words
from modules.modules_svm.preprocessing_svm import map_one_hot_to_labels
from modules.utils import setup_logging

import os
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.metrics import classification_report, multilabel_confusion_matrix

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def train_svm(df, base_path, vectorizer="tf-idf", use_cached_embeddings=False, emb_caching_path="code/models/multilingual_embeddings.npy"):
    """
    Train an SVM model for multiclass multilabel classification.
    
    Args:
        df: DataFrame containing the training data.
        base_path: Base path for saving model outputs.
        vectorizer: Type of vectorizer to use (tf-idf, bag-of-words, multilingual-embeddings).
        use_cached_embeddings: Whether to use cached embeddings for multilingual-embeddings.
        emb_caching_path: Path to save/load cached embeddings.
    
    Returns:
        dict: Classification report.
    """
    try:
        # Create label mapping
        all_narratives = df['narrative_subnarrative_pairs'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        ).tolist()
        label_mapping = create_label_mapping(all_narratives)
        
        # Prepare data
        texts, labels, label_mapping, filenames = prepare_data(df, label_mapping)

        # shorten texts and labels to 15 instances for testing
        # texts = texts[:5]
        # labels = labels[:5]        

        if vectorizer == "tf-idf":
            # Vectorize text using TF-IDF
            vectorizer = TfidfVectorizer(max_features=5000)
            X = vectorizer.fit_transform(texts)
        elif vectorizer == "bag-of-words":
            # Vectorize text using Bag of Words
            X = bag_of_words(texts)    
        elif vectorizer == "multilingual-embeddings":
            if use_cached_embeddings and os.path.exists(os.path.join(base_path, emb_caching_path)):
                X = np.load(os.path.join(base_path, emb_caching_path))
            else:
                # Load model directly
                tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-uncased")
                model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-multilingual-uncased", output_hidden_states=True)
                X = []
                for text in tqdm(texts, desc="Calculating embeddings"):
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                    outputs = model(**inputs)
                    embeddings = outputs.hidden_states[-1].mean(dim=1).detach().numpy()
                    X.append(embeddings)
                X = np.vstack(X)
                
                # Save embeddings
                np.save(os.path.join(base_path, emb_caching_path), X)
        else:
            raise ValueError(f"Unknown vectorizer: {vectorizer}")


        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
            X, labels, filenames, test_size=0.2, random_state=42,
        )

        # Train SVM using OneVsRestClassifier
        model = OneVsRestClassifier(LinearSVC())
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        
        ## The following code for creating df_with_predictions is for qualitative analysis funcitonality
        # Reverse the label mapping for easier lookup
        reverse_label_mapping = {v: eval(k) for k, v in label_mapping.items()}
        
        # Map predictions and true labels to lists of dictionaries
        predicted_labels_mapped = map_one_hot_to_labels(y_pred, reverse_label_mapping)
        true_labels_mapped = map_one_hot_to_labels(y_test, reverse_label_mapping)
        
        # Map predictions back to the original DataFrame
        predictions_df = pd.DataFrame({
            'filename': filenames_test,
            'predicted_labels': predicted_labels_mapped,
            'true_labels': true_labels_mapped,
        })
        df_with_predictions = pd.merge(df, predictions_df, on='filename', how='left')
        
        # Drop rows from train set
        df_with_predictions = df_with_predictions.dropna(subset=['predicted_labels'])
        df_with_predictions.reset_index(drop=True, inplace=True)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        #logger.info("Training classification report:")
        #logger.info(json.dumps(report, indent=2))


        # count the occurrences of each narrative in the original data, training set, test set, and predictions for later analysis
        flat_narratives = [item for sublist in df['narrative_subnarrative_pairs'].apply(eval) for item in sublist]
        flat_labels = []
        for narrative in flat_narratives:
            narrative_str = str(narrative)
            flat_labels.append(label_mapping[narrative_str])
        onehot = pd.get_dummies(pd.Series(flat_labels)).to_numpy()
        onehot = (onehot > 0).astype(int)

        idx_to_label = {v: k for k, v in label_mapping.items()}

        counts_original = pd.DataFrame(onehot).sum(axis=0).to_numpy()
        counts_train = pd.DataFrame(y_train).sum(axis=0).to_numpy()
        counts_test = pd.DataFrame(y_test).sum(axis=0).to_numpy()
        counts_df = pd.DataFrame({
            'Narrative Index': range(len(counts_original)),
            'Occurrences in Original DF': counts_original,
            'Occurrences in Train Set': counts_train,
            'Occurrences in Test Set': counts_test,
            'Occurrences in Predictions': [np.sum(y_pred[:, i]) for i in range(len(counts_original))],
            'Narrative': [idx_to_label[i] for i in range(len(counts_original))]
        })

        # compute confusion matrix for later analysis
        confusion_mtx = multilabel_confusion_matrix(y_test, y_pred)

        return report, counts_df, confusion_mtx, df_with_predictions

    except Exception as e:
        raise logger.error(f"Error in SVM training: {str(e)}")