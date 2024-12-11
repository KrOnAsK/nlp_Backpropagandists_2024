# modules/text_segmentation.py

import nltk
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def tokenize_text(df):
    """
    Tokenize text content into sentences and words
    """
    try:
        nltk.download('punkt', quiet=True)
        
        df['tokens'] = None
        for i, row in df.iterrows():
            # Split content into sentences
            sentences = nltk.sent_tokenize(row['content'])
            # Tokenize each sentence
            tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
            df.at[i, 'tokens'] = tokens
            
        return df
    
    except Exception as e:
        logger.error(f"Error in tokenize_text: {str(e)}")
        raise

def handle_unusual_sentences(df):
    """
    Handle sentences of unusual length through merging or splitting
    """
    try:
        def merge_specific_sentence(sentences, sentence_idx, direction):
            if direction == 'previous':
                merged = sentences[sentence_idx - 1] + sentences[sentence_idx]
                sentences[sentence_idx - 1] = merged
                del sentences[sentence_idx]
            elif direction == 'next':
                merged = sentences[sentence_idx] + sentences[sentence_idx + 1]
                sentences[sentence_idx] = merged
                del sentences[sentence_idx + 1]
            return sentences
        
        # Handle specific cases that need merging
        merge_cases = [
            (87, 6, 'previous'),
            (88, 0, 'next'),
            (127, 23, 'next'),
            (136, 3, 'next')
        ]
        
        for row_idx, sentence_idx, direction in merge_cases:
            if row_idx in df.index:
                sentences = df.at[row_idx, 'tokens']
                df.at[row_idx, 'tokens'] = merge_specific_sentence(sentences, sentence_idx, direction)
        
        # Additional sentence handling logic can be added here
        
        return df
    
    except Exception as e:
        logger.error(f"Error in handle_unusual_sentences: {str(e)}")
        raise