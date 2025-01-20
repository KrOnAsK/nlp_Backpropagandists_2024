import os
import pandas as pd
import wandb
import torch
import logging
from datetime import datetime
from huggingface_hub import login

from model import initialize_model, setup_peft
from data_utils import prepare_data
from trainer import train_model
from debug_utils import debug_misclassifications
from modules.utils import compute_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Login to Hugging Face
        login('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')

        # Check CUDA availability
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Set paths
        # Find repository root by looking for .git directory
        def find_repo_root():
            current = os.getcwd()
            while current != os.path.dirname(current):  # while we haven't hit the root directory
                if os.path.exists(os.path.join(current, '.git')):
                    return current
                current = os.path.dirname(current)
            raise Exception("No .git directory found - repository root could not be determined")

        # Set paths using repository root
        repo_root = find_repo_root()
        code_path = os.path.join(repo_root, "code")
        current_date = datetime.now().strftime("%Y%m%d")
        output_dir = os.path.join(code_path, "models", f"llama_{current_date}")
        os.makedirs(output_dir, exist_ok=True)

        # Load data from code directory
        print("\nLoading datasets...")
        print(f"Repository root: {repo_root}")
        print(f"Looking for data files in: {code_path}")
        input_file_full = os.path.join(code_path, "df_normalized.csv")
        input_file_ua = os.path.join(code_path, "df_normalized_ua.csv")
        input_file_cc = os.path.join(code_path, "df_normalized_cc.csv")

        df_normalized = pd.read_csv(input_file_full)
        df_normalized_ua = pd.read_csv(input_file_ua)
        df_normalized_cc = pd.read_csv(input_file_cc)

        # Display dataset information
        print("\nFull Dataset Info:")
        print(df_normalized.info())
        print(f"\nNumber of records: {len(df_normalized)}")

        print("\nUA Dataset Info:")
        print(df_normalized_ua.info())
        print(f"\nNumber of UA records: {len(df_normalized_ua)}")

        print("\nCC Dataset Info:")
        print(df_normalized_cc.info())
        print(f"\nNumber of CC records: {len(df_normalized_cc)}")

        # Choose dataset
        print("\nSelect dataset for training:")
        print("1. Full dataset")
        print("2. UA dataset")
        print("3. CC dataset")
        choice = input("Enter your choice (1-3): ")

        if choice == "1":
            print("\nTraining on full dataset...")
            selected_df = df_normalized
        elif choice == "2":
            print("\nTraining on UA dataset...")
            selected_df = df_normalized_ua
        else:
            print("\nTraining on CC dataset...")
            selected_df = df_normalized_cc

        # Initialize wandb
        wandb.init(project="llama-classification", name=f"llama-classification-{current_date}")

        # Model configuration
        model_name = "openlm-research/open_llama_7b"

        # Prepare data
        train_dataset, val_dataset, tokenizer, label_mapping, num_labels = prepare_data(
            selected_df, model_name, output_dir
        )

        # Initialize and setup model
        print("\nInitializing model...")
        model = initialize_model(model_name, num_labels)
        model = setup_peft(model)

        # Train model
        trainer = train_model(model, train_dataset, val_dataset, output_dir, current_date)

        # Evaluate model
        print("\nEvaluating model...")
        results = trainer.evaluate()
        
        print("\nEvaluation results:")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

        # Save model and tokenizer
        print("\nSaving model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Debug misclassifications
        misclass_df = debug_misclassifications(selected_df, model, tokenizer, label_mapping)

        # End wandb run
        wandb.finish()

        return results, model, tokenizer, label_mapping

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        raise

if __name__ == "__main__":
    main()