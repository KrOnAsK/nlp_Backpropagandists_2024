import os
import pandas as pd
from modules.data_loader import load_initial_data
from modules.text_segmentation import tokenize_text, handle_unusual_sentences
from modules.text_normalization import normalize_text
from modules.connlu_converter import convert_to_connlu
from modules.utils import setup_logging
from dl_methods.transformer import train_bert
import logging
from non_dl_methods.naive_bayes import *
import pickle

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Define paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    documents_path = os.path.join(base_path, "../training_data_16_October_release/EN/raw-documents")
    annotations_file = os.path.join(base_path, "../training_data_16_October_release/EN/subtask-2-annotations.txt")
    output_dir = os.path.join(base_path, "../CoNLL")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Load and prepare initial data
        logger.info("Loading initial data...")
        df = load_initial_data(documents_path, annotations_file)
        logger.info(f"Loaded {len(df)} documents")

        # 2. Tokenize text
        logger.info("Tokenizing text...")
        df = tokenize_text(df)
        
        # 3. Handle unusual sentences
        logger.info("Handling unusual sentences...")
        df = handle_unusual_sentences(df)
        
        # 4. Normalize text
        logger.info("Normalizing text...")
        df = normalize_text(df)
        # print(df.head())
        # print(df.columns)
        # print(type(df['tokens_normalized'].iloc[0]))
        # print(df['tokens_normalized'].iloc[0])
        # print(df['narrative_subnarrative_pairs'].iloc[0])
        # 5. Convert to CoNLL-U format
        # only use when ConLL-U format is needed
        #logger.info("Converting to CoNLL-U format...")
        #convert_to_connlu(df, output_dir, 'tokens')
        logger.info("Preprocessing completed successfully")

        
        # 6. Train BERT model
        logger.info("Starting BERT training...")
        training_results = train_bert(df, base_path)
        logger.info(f"BERT training completed. Results: {training_results}")
        
        
        
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {str(e)}")
        raise

    return df

if __name__ == "__main__":
    
    # cache preprocessed data using pickle
    df = None
    try:
        with open('df_preprocessed_cache.pkl', 'rb') as f:
            df = pickle.load(f)
            print("Loaded preprocessed data from cache")
    except:
        print("Preprocessed data not found in cache. Running preprocessing...")
        df = main()
        with open('df_preprocessed_cache.pkl', 'wb') as f:
            pickle.dump(df, f)    

    nb_results = run_non_dl_method(df, "naive_bayes", "bag_of_words")
    print("Results from Naive Bayes: ", nb_results)
