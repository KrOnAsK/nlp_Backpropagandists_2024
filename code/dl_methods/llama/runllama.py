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
from modules.utils import debug_misclassifications, setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_single_dataset(df, model_name, output_dir, current_date, dataset_name):
    """
    Train model on a single dataset

    Args:
        df: DataFrame containing the dataset
        model_name: Name of the model to use
        output_dir: Directory to save outputs
        current_date: Current date string for naming
        dataset_name: Name of the dataset for logging

    Returns:
        tuple: (results, model, tokenizer, label_mapping)
    """
    # Create dataset-specific output directory
    dataset_output_dir = os.path.join(output_dir, f"{dataset_name}_{current_date}")
    os.makedirs(dataset_output_dir, exist_ok=True)

    print(f"\nTraining on {dataset_name} dataset...")

    # Initialize wandb run for this dataset
    wandb.init(
        project="llama-classification",
        name=f"llama-classification-{dataset_name}-{current_date}",
        reinit=True,
    )

    # Prepare data
    train_dataset, val_dataset, tokenizer, label_mapping, num_labels = prepare_data(
        df, model_name, dataset_output_dir
    )

    # Initialize and setup model
    print("\nInitializing model...")
    model = initialize_model(model_name, num_labels)
    model = setup_peft(model)

    # Train model
    trainer = train_model(
        model,
        train_dataset,
        val_dataset,
        dataset_output_dir,
        current_date,
        dataset_name,
    )

    # Evaluate model
    print("\nEvaluating model...")
    results = trainer.evaluate()

    print(f"\nEvaluation results for {dataset_name} dataset:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Save model and tokenizer
    print(f"\nSaving {dataset_name} model...")
    trainer.save_model(dataset_output_dir)
    tokenizer.save_pretrained(dataset_output_dir)

    # Debug misclassifications
    misclass_df = debug_misclassifications(df, model, tokenizer, label_mapping)

    # End wandb run
    wandb.finish()

    return results, model, tokenizer, label_mapping


def main():
    try:
        # Login to Hugging Face
        login("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")

        # Check CUDA availability
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )

        # Set paths
        def find_repo_root():
            current = os.getcwd()
            while current != os.path.dirname(current):
                if os.path.exists(os.path.join(current, ".git")):
                    return current
                current = os.path.dirname(current)
            raise Exception(
                "No .git directory found - repository root could not be determined"
            )

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

        # Model configuration
        model_name = "openlm-research/open_llama_7b"

        # Choose dataset
        print("\nSelect training mode:")
        print("1. Full dataset only")
        print("2. UA dataset only")
        print("3. CC dataset only")
        print("4. Sequential training (UA -> CC -> Full)")
        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            results, model, tokenizer, label_mapping = train_single_dataset(
                df_normalized, model_name, output_dir, current_date, "full"
            )
        elif choice == "2":
            results, model, tokenizer, label_mapping = train_single_dataset(
                df_normalized_ua, model_name, output_dir, current_date, "ua"
            )
        elif choice == "3":
            results, model, tokenizer, label_mapping = train_single_dataset(
                df_normalized_cc, model_name, output_dir, current_date, "cc"
            )
        else:
            print("\nStarting sequential training...")

            # Train on UA dataset
            ua_results, ua_model, ua_tokenizer, ua_label_mapping = train_single_dataset(
                df_normalized_ua, model_name, output_dir, current_date, "ua"
            )

            # Train on CC dataset
            cc_results, cc_model, cc_tokenizer, cc_label_mapping = train_single_dataset(
                df_normalized_cc, model_name, output_dir, current_date, "cc"
            )

            # Train on full dataset
            results, model, tokenizer, label_mapping = train_single_dataset(
                df_normalized, model_name, output_dir, current_date, "full"
            )

            print("\nSequential training completed!")

        return results, model, tokenizer, label_mapping

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback

        traceback.print_exc()
        wandb.finish()
        raise


if __name__ == "__main__":
    main()
