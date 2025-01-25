from collections import defaultdict
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
from modules.modules_svm.preprocessing_svm import create_label_mapping

def extract_metrics(training_results):
    """
    Extract the precision, recall, and F1-score from the training results.

    Args:
        training_results: Dictionary containing the training results.

    Returns:
        dict: Dictionary containing the metrics.
    """

    metrics = {
        'macro avg': training_results['macro avg'],
        'weighted avg': training_results['weighted avg']
    }
    return metrics

def create_metrics_dataframe(datasets, methods, results):
    """
    Create a DataFrame to hold the metrics for specified datasets and methods, which facilitates comparison.
    
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
    """
    Show counts_df as a styled table. Counts_df should contain the class occurrences in train, test, and predictions.

    Args:
        counts_df: DataFrame containing the class occurrences
    """

    display(counts_df.style.set_table_styles(
        [{'selector': 'thead th', 'props': [('background-color', '#f7f7f9'), ('color', 'black')]}]
    ).set_properties(**{'text-align': 'center'}).set_caption("Table of Class Occurrences in Train, Test, and Predictions"))

def show_confusion_matrix(df, confusion_mtx):
    """
    Visualize the confusion matrix for each narrative-subnarrative pair.
    
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

def sort_narratives_recall(confusion_matrix, threshold_good=0.4, threshold_bad=0.1):
    # find narratives with recall > 0.5 as good results and narratives with recall <= 0.2 as bad results
    good_narratives = []
    bad_narratives = []

    # Extract recall values and sort by recall
    recall_values = [(i, confusion_matrix[i][1][1] / (confusion_matrix[i][1][1] + confusion_matrix[i][1][0])) for i in range(len(confusion_matrix)) if confusion_matrix[i][1][1] + confusion_matrix[i][1][0] >= 5]
    sorted_narratives = sorted(recall_values, key=lambda x: x[1], reverse=True)

    # Assign to good_narratives and bad_narratives arrays
    for i, recall in sorted_narratives:
        if recall > threshold_good and len(good_narratives) < 5:
            good_narratives.append(i)
        elif recall < threshold_bad and len(bad_narratives) < 5:
            bad_narratives.append(i)
        if len(good_narratives) >= 5 and len(bad_narratives) >= 5:
            break

    return good_narratives, bad_narratives, sorted_narratives

def plot_difference_barchart(X, labels, sorted_narratives, label_mapping):
    """
    Plot barchart of absolute difference between average vector of narrative and average vector of instances not belonging to it.
    
    Parameters:
        X (np.array): Feature matrix.
        labels (np.array): Label matrix.
        sorted_narratives (list): List of tuples containing narrative index and recall value.
        label_mapping (dict): Mapping of narrative index to narrative string.
    """
    diff_vectors = []
    recall_values = []
    support_values = []
    for i, recall in sorted_narratives:
        data_indices = [j for j, label in enumerate(labels) if label[i] == 1]
        non_data_indices = [j for j, label in enumerate(labels) if label[i] == 0]
        X_data = X[data_indices]
        X_non_data = X[non_data_indices]
        avg_vector_data = np.asarray(np.mean(X_data, axis=0))
        avg_vector_non_data = np.asarray(np.mean(X_non_data, axis=0))
        diff_vector_data = np.linalg.norm(avg_vector_non_data - avg_vector_data)
        diff_vectors.append(diff_vector_data)
        recall_values.append(recall)
        support_values.append(len(data_indices))

    # x tick labels: narratives
    x_tick_labels = []
    for idx, _ in sorted_narratives:
        narrative = list(label_mapping.keys())[list(label_mapping.values()).index(idx)]
        narrative_dict = eval(narrative)
        narrative_str = f"{narrative_dict['narrative'][:5]}-{narrative_dict['subnarrative'][:5]}[{idx}]"
        x_tick_labels.append(narrative_str)

    fig, ax1 = plt.subplots(figsize=(15, 5))

    ax1.bar(x_tick_labels, diff_vectors, color='b', alpha=0.6)
    ax1.set_xlabel('Narratives')
    ax1.set_ylabel('Sum of absolute difference', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticklabels(x_tick_labels, rotation=45, ha='right')

    ax2 = ax1.twinx()
    ax2.plot(x_tick_labels, recall_values, color='r', marker='o', linestyle='-', linewidth=2, markersize=5)
    ax2.set_ylabel('Recall', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(x_tick_labels, support_values, color='g', marker='x', linestyle='-', linewidth=2, markersize=5)
    ax3.set_ylabel('Support', color='g')
    ax3.tick_params(axis='y', labelcolor='g')

    plt.title("Difference between (avg) narrative vector and (avg) vector over whole dataset.")
    plt.show()
    
def plot_differences_heatmap(X, labels, label_mapping, vectorizer, narratives=[]):
    """
    Plot heatmap of the 100 most important words for selected narratives.

    Parameters:
        X (np.array): Feature matrix.
        labels (np.array): Label matrix.
        label_mapping (dict): Mapping of narrative index to narrative string.
        vectorizer (TfidfVectorizer): TfidfVectorizer object.
        narratives (list): List of narrative indices to plot heatmap for.
    """
    diff_vectors = []
    for i in narratives:
        data_indices = [j for j, label in enumerate(labels) if label[i] == 1]
        non_data_indices = [j for j, label in enumerate(labels) if label[i] == 0]
        X_data = X[data_indices]
        X_non_data = X[non_data_indices]
        avg_vector_data = np.asarray(np.mean(X_data, axis=0))
        avg_vector_non_data = np.asarray(np.mean(X_non_data, axis=0))
        diff_vector_data = abs(avg_vector_non_data - avg_vector_data)
        diff_vectors.append(diff_vector_data)

    diff_vector_combined = np.concatenate(diff_vectors, axis=0)

    # Get the 100 most important words without sorting
    feature_names = vectorizer.get_feature_names_out()
    importance_indices = np.argsort(np.sum(diff_vector_combined, axis=0))[-100:]
    importance_indices = sorted(importance_indices)  # Keep the original order
    diff_vector_combined = diff_vector_combined[:, importance_indices]
    x_tick_labels = feature_names[importance_indices]

    # y tick labels left: narratives
    y_tick_labels = []
    for idx in narratives:
        narrative = list(label_mapping.keys())[list(label_mapping.values()).index(idx)]
        narrative_dict = eval(narrative)
        narrative_str = f"{narrative_dict['narrative'][:5]}-{narrative_dict['subnarrative'][:5]}[{idx}]"
        y_tick_labels.append(narrative_str)

    plt.figure(figsize=(20, 5))
    ax = sns.heatmap(diff_vector_combined, cmap='coolwarm', center=0, xticklabels=x_tick_labels, yticklabels=y_tick_labels)
    ax.yaxis.set_ticks_position('left')
    plt.title("Heatmap of 100 most important words for narrative(s): " + ", ".join(y_tick_labels))
    plt.show()
