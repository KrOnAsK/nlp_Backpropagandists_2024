# modules/text_normalization.py

import nltk
import re
import string
import stanza
import torch
from nltk.corpus import stopwords
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def normalize_text(df, column_name='tokens'):
    """
    Normalize text through lowercasing, removing stopwords, punctuation, and lemmatization
    """
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize Stanza
        nlp = stanza.Pipeline('en',
                            processors='tokenize,lemma',
                            device=device,
                            use_gpu=torch.cuda.is_available(),
                            batch_size=4096,
                            tokenize_batch_size=4096,
                            tokenize_pretokenized=True,
                            download_method=None)
        
        # Get stopwords
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        
        def normalize_token(token):
            if not isinstance(token, str):
                return ''
            # Convert to lowercase and strip
            token = token.lower().strip()
            # Remove special characters and numbers
            token = re.sub(r'[^a-z]', '', token)
            return token
        
        def clean_text(nested_tokens):
            if not isinstance(nested_tokens, list):
                return []
            
            cleaned_tokens = []
            for sentence in nested_tokens:
                if isinstance(sentence, list):
                    for token in sentence:
                        normalized = normalize_token(token)
                        if normalized and normalized not in stop_words:
                            cleaned_tokens.append(normalized)
            
            return cleaned_tokens
        
        def process_text(tokens):
            try:
                if not tokens:
                    return []
                text = ' '.join(tokens)
                doc = nlp(text)
                lemmas = []
                for sent in doc.sentences:
                    for word in sent.words:
                        lemma = word.lemma.lower()
                        if lemma not in stop_words:
                            lemmas.append(lemma)
                return lemmas
            except Exception as e:
                logger.error(f"Error processing text: {str(e)}")
                return []
        
        def process_batch(batch_tokens):
            results = []
            for tokens in batch_tokens:
                cleaned_tokens = clean_text(tokens)
                normalized = process_text(cleaned_tokens)
                normalized = [normalize_token(token) for token in normalized if normalize_token(token)]
                results.append(normalized)
            return results
        
        # Process in batches
        batch_size = 50
        normalized_tokens = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        logger.info(f"Starting processing of {len(df)} rows in {total_batches} batches")
        
        for i in tqdm(range(0, len(df), batch_size), desc="Normalizing text"):
            batch_df = df.iloc[i:i + batch_size]
            batch_tokens = batch_df[column_name].tolist()
            normalized_batch = process_batch(batch_tokens)
            normalized_tokens.extend(normalized_batch)
        
        # Update DataFrame
        df_normalized = df.copy()
        df_normalized[f'{column_name}_normalized'] = normalized_tokens
        
        return df_normalized
    
    except Exception as e:
        logger.error(f"Error in normalize_text: {str(e)}")
        raise