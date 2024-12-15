import os
import pandas as pd
from datetime import datetime
from modules.data_loader import load_initial_data
from modules.text_segmentation import tokenize_text, handle_unusual_sentences
from modules.text_normalization import normalize_text
from modules.connlu_converter import convert_to_connlu
from modules.utils import setup_logging
from dl_methods.transformer import train_bert, debug_misclassifications
import logging

class ProcessingSummary:
    def __init__(self):
        self.start_time = datetime.now()
        self.steps_completed = []
        self.ml_results = {}
        self.document_stats = {}
        
    def add_step(self, step_name, details=None):
        step = {
            'name': step_name,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'details': details
        }
        self.steps_completed.append(step)
    
    def add_ml_result(self, model_type, metrics):
        self.ml_results[model_type] = metrics
    
    def display_summary(self):
        duration = datetime.now() - self.start_time
        minutes = int(duration.total_seconds() // 60)
        seconds = int(duration.total_seconds() % 60)
        
        # Print header
        print("\n" + "="*80)
        print(f"{'PROCESSING SUMMARY':^80}")
        print("="*80)
        
        # Print general statistics
        print("\nGENERAL INFORMATION")
        print("-"*80)
        print(f"Total Processing Time: {minutes}m {seconds}s")
        print(f"Steps Completed: {len(self.steps_completed)}")
        
        # Print document statistics
        print("\nDOCUMENT STATISTICS")
        print("-"*80)
        for key, value in self.document_stats.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Print processing timeline
        print("\nPROCESSING TIMELINE")
        print("-"*80)
        for step in self.steps_completed:
            print(f"\n[{step['timestamp']}] {step['name']}")
            if step.get('details'):
                for key, value in step['details'].items():
                    print(f"  └─ {key}: {value}")
        
        # Print ML results if any
        if self.ml_results:
            print("\nML RESULTS")
            print("-"*80)
            for model, metrics in self.ml_results.items():
                print(f"\n{model}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  └─ {metric}: {value:.4f}")
                    else:
                        print(f"  └─ {metric}: {value}")
        
        print("\n" + "="*80 + "\n")

def get_ml_choice():
    print("\nSelect processing option:")
    print("1. Train BERT on all data")
    print("2. Train BERT on UA data only")
    print("3. Train BERT on CC data only")
    print("4. Run all BERT training variations")
    print("5. Convert to CoNLL-U format")
    print("6. Skip additional processing")
    print("6. Debugging UA")
    print("6. Debugging CC")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-8): "))
            if 1 <= choice <= 8:
                return choice
            print("Please enter a number between 1 and 8.")
        except ValueError:
            print("Please enter a valid number.")

def run_selected_ml(choice, df_normalized, df_normalized_ua, df_normalized_cc, df, base_path, output_dir, logger, summary):
    if choice == 1:
        logger.info("Starting BERT training on full dataset...")
        training_results = train_bert(df_normalized, base_path)
        summary.add_ml_result('BERT (Full Dataset)', training_results)
        logger.info(f"BERT training on full data completed. Results: {training_results}")
    
    elif choice == 2:
        logger.info("Starting BERT training on UA dataset...")
        training_results_ua = train_bert(df_normalized_ua, base_path)
        summary.add_ml_result('BERT (UA Dataset)', training_results_ua)
        logger.info(f"BERT training on UA data completed. Results: {training_results_ua}")
    
    elif choice == 3:
        logger.info("Starting BERT training on CC dataset...")
        training_results_cc = train_bert(df_normalized_cc, base_path)
        summary.add_ml_result('BERT (CC Dataset)', training_results_cc)
        logger.info(f"BERT training on CC data completed. Results: {training_results_cc}")
    
    elif choice == 4:
        logger.info("Starting BERT training on all variations...")
        training_results = train_bert(df_normalized, base_path)
        summary.add_ml_result('BERT (Full Dataset)', training_results)
        logger.info(f"BERT training on full data completed. Results: {training_results}")
        
        training_results_ua = train_bert(df_normalized_ua, base_path)
        summary.add_ml_result('BERT (UA Dataset)', training_results_ua)
        logger.info(f"BERT training on UA data completed. Results: {training_results_ua}")
        
        training_results_cc = train_bert(df_normalized_cc, base_path)
        summary.add_ml_result('BERT (CC Dataset)', training_results_cc)
        logger.info(f"BERT training on CC data completed. Results: {training_results_cc}")
    
    elif choice == 5:
        logger.info("Converting to CoNLL-U format...")
        convert_to_connlu(df, output_dir, 'tokens')
        summary.add_step('CoNLL-U Conversion', {'output_directory': output_dir})
        logger.info("CoNLL-U conversion completed successfully")

    elif choice == 7:
        logger.info("Debugging missclassified words with BERT training on UA dataset...")
        debugging_ua = debug_misclassifications(df_normalized_ua, dataset_type="Training", min_examples_per_class=1)
        print(debugging_ua)
        logger.info(f"BERT training on UA data completed. Results: {debugging_ua}")

    elif choice == 8:
        logger.info("Debugging missclassified words with BERT training on CC dataset...")
        debugging_cc = debug_misclassifications(df_normalized_cc, dataset_type="Training", min_examples_per_class=1)
        logger.info(f"BERT training on CC data completed. Results: {debugging_cc}")


def main():
    # Initialize processing summary
    summary = ProcessingSummary()
    
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
        summary.add_step('Data Loading', {'document_count': len(df)})
        logger.info(f"Loaded {len(df)} documents")

        # 2. Tokenize text
        logger.info("Tokenizing text...")
        df = tokenize_text(df)
        summary.add_step('Text Tokenization')
        
        # 3. Handle unusual sentences
        logger.info("Handling unusual sentences...")
        df = handle_unusual_sentences(df)
        summary.add_step('Unusual Sentence Handling')
        
        # 4. Normalize text
        logger.info("Normalizing text...")
        df_normalized, df_normalized_ua, df_normalized_cc = normalize_text(df)
        summary.add_step('Text Normalization', {
            'total_documents': len(df_normalized),
            'ua_documents': len(df_normalized_ua),
            'cc_documents': len(df_normalized_cc)
        })
        print(df_normalized_ua.columns)
        logger.info("Preprocessing completed successfully")
        
        # Document statistics
        summary.document_stats = {
            'total_documents': len(df_normalized),
            'ua_documents': len(df_normalized_ua),
            'cc_documents': len(df_normalized_cc),
            'average_tokens_per_doc': df['tokens'].str.len().mean() if 'tokens' in df.columns else None
        }

        # Get user input for ML approach
        choice = get_ml_choice()
        
        # Run selected ML approach
        if choice != 6:
            run_selected_ml(choice, df_normalized, df_normalized_ua, df_normalized_cc, df, base_path, output_dir, logger, summary)
        else:
            logger.info("Additional processing skipped as per user choice")
            summary.add_step('Processing Skipped')
        
        # Display summary
        summary.display_summary()
        
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {str(e)}")
        summary.add_step('Error', {'error_message': str(e)})
        raise

if __name__ == "__main__":
    main()