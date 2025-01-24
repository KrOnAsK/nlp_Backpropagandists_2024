from collections import defaultdict
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
from modules.modules_svm.preprocessing_svm import create_label_mapping

def extract_metrics(training_results):
    metrics = {
        'macro avg': training_results['macro avg'],
        'weighted avg': training_results['weighted avg']
    }
    return metrics

def create_metrics_dataframe(datasets, methods, results):
    """
    Create a DataFrame to hold all metrics for given datasets and methods.
    
    Args:
        datasets: List of dataset names.
        methods: List of method names.
        results: Dictionary containing the results for each method and dataset.
    
    Returns:
        pd.DataFrame: DataFrame containing the metrics.
    """
    metrics_df = pd.DataFrame()

    for method in methods:
        for dataset in datasets:
            result_var = f'{method}_{dataset}'
            metrics = extract_metrics(results[result_var])
            for metric_type, values in metrics.items():
                metrics_df = pd.concat([metrics_df, pd.DataFrame([{
                    'Method': method,
                    'Dataset': dataset,
                    'Metric': metric_type,
                    'Precision': round(values['precision'], 3),
                    'Recall': round(values['recall'], 3),
                    'F1-Score': round(values['f1-score'], 3)
                }])], ignore_index=True)

    # Pivot the DataFrame for better visualization
    metrics_pivot = metrics_df.pivot_table(index=['Method', 'Dataset'], columns='Metric', values=['Precision', 'Recall', 'F1-Score'])

    # Format the DataFrame to show only three digits
    metrics_pivot = metrics_pivot.applymap(lambda x: '{:.3f}'.format(x))

    # Highlight the weighted avg columns
    highlight = lambda x: ['border-left: 4px solid darkblue; border-right: 4px solid darkblue' if 'weighted avg' in col else '' for col in x.index]

    # Display the table
    return metrics_pivot.style.apply(highlight, axis=1).set_table_styles(
        [{'selector': 'thead th', 'props': [('background-color', '#f7f7f9'), ('color', 'black')]}]
    ).set_properties(**{'text-align': 'center'}).set_caption("Performance Metrics for Each Method and Dataset")
    
def show_class_occurrences(counts_df):
    # Display the DataFrame as a table
        display(counts_df.style.set_table_styles(
            [{'selector': 'thead th', 'props': [('background-color', '#f7f7f9'), ('color', 'black')]}]
        ).set_properties(**{'text-align': 'center'}).set_caption("Table of Class Occurrences in Train, Test, and Predictions"))

def show_confusion_matrix(df, confusion_mtx): ### TRACK CHANGE added df
    """
    Show confusion matrix for SVM model.
    
    Args:
        confusion_mtx: Multilabel confusion matrix.
    """
    n_classes = len(confusion_mtx)
    ncols = 6 
    nrows = math.ceil(n_classes / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 2.5))
    axes = axes.flatten()

    all_narratives = df['narrative_subnarrative_pairs'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        ).tolist()
    label_mapping = create_label_mapping(all_narratives)

    # Plot each confusion matrix
    for idx, mtx in enumerate(confusion_mtx):
        sns.heatmap(mtx, annot=True, fmt='d', ax=axes[idx], cmap='Blues')    
        narrative = list(label_mapping.keys())[list(label_mapping.values()).index(idx)]
        narrative_dict = eval(narrative)
        narrative_str = f"{narrative_dict['narrative'][:5]}-{narrative_dict['subnarrative'][:5]}[{idx}]"
        axes[idx].set_title(narrative_str)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')

    # Hide empty subplots
    for idx in range(n_classes, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

def show_accumulated_confusion_matrix(confusion_mtx):
    """
    Show a matrix of accumulated confusion matrices.
    
    Args:
        confusion_mtx: Multilabel confusion matrix.
    """

    # Sum all confusion matrices
    accumulated_confusion = np.sum(confusion_mtx, axis=0)

# Function to extract true positives and false negatives for specific class for qualitative analysis
def extract_confusion_data(df, target_label):
    # Initialize lists to store results
    false_negatives = [] 
    true_positives = []  
    
    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        true_labels = row['true_labels']  # Extract true labels
        predicted_labels = row['predicted_labels']  # Extract predicted labels
        
        # Extract the indices of true and predicted labels matching the target
        true_indices = [
            label[1] for label in true_labels
            if label[1] == target_label  # Match target narrative and subnarrative
        ]
        predicted_indices = [
            label[1] for label in predicted_labels
            if label[1] == target_label  # Match target narrative and subnarrative
        ]
        
        # Determine false negatives (true label exists, but no matching prediction)
        if true_indices and not any(idx in true_indices for idx in predicted_indices):
            false_negatives.append(row)
        
        # Determine true positives (matching true and predicted labels)
        if any(idx in true_indices for idx in predicted_indices):
            true_positives.append(row)
    
    return false_negatives, true_positives

def find_classes_with_tp_and_fn_language_check(df):
    """
    Identify class indices where there is at least one true positive and one false negative,
    and both must have "EN" in the language column.
    
    Parameters:
        df (DataFrame): Input DataFrame containing `true_labels`, `predicted_labels`, and `language`.
    
    Returns:
        classes_with_tp_fn (list): List of class indices that meet the condition.
    """

    # Dictionary to store counts of true positives and false negatives for each class index
    class_stats = defaultdict(lambda: {"tp": 0, "fn": 0})
    
    # Iterate over all rows in the DataFrame
    for index, row in df.iterrows():
        true_labels = row['true_labels']  # Extract true labels
        predicted_labels = row['predicted_labels']  # Extract predicted labels
        language = row['language']  # Extract language

        # Skip rows that are not in "EN"
        if language != "EN":
            continue
        
        # Get all unique class indices in true and predicted labels
        true_indices = [label[1] for label in true_labels]
        predicted_indices = [label[1] for label in predicted_labels]
        
        # Check for true positives (class exists in both true and predicted labels)
        for idx in true_indices:
            if idx in predicted_indices:
                class_stats[idx]["tp"] += 1
        
        # Check for false negatives (class exists in true but not in predicted labels)
        for idx in true_indices:
            if idx not in predicted_indices:
                class_stats[idx]["fn"] += 1
    
    # Filter classes that have at least one true positive and one false negative
    classes_with_tp_fn = [
        class_idx for class_idx, stats in class_stats.items() 
        if stats["tp"] > 0 and stats["fn"] > 0
    ]
    
    return classes_with_tp_fn