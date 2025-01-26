import logging
from modules.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def create_label_mapping(all_narratives):
    """
    Create a consistent mapping for all narrative pairs
    
    Args:
        all_narratives: List of lists of narrative dictionaries
    
    Returns:
        dict: Mapping from narrative string to numeric index
    """
    unique_narratives = set()
    for narratives in all_narratives:
        for narrative in narratives:
            narrative_str = str(narrative) 
            unique_narratives.add(narrative_str)
    
    # Create mapping
    narrative_to_idx = {
        narrative: idx 
        for idx, narrative in enumerate(sorted(unique_narratives))
    }
    
    logger.info(f"Created mapping for {len(narrative_to_idx)} unique narratives")
    return narrative_to_idx

def get_first_narrative_label(narrative_list, label_mapping):
    """
    Convert first narrative in list to numeric label
    
    Args:
        narrative_list: List of narrative dictionaries
        label_mapping: Dictionary mapping narrative strings to indices
    
    Returns:
        int: Numeric label for the first narrative
    """
    if narrative_list and len(narrative_list) > 0:
        narrative_str = str(narrative_list[0])
        return label_mapping[narrative_str]
    return None

def prepare_data(df, label_mapping=None):
    """
    Prepare data for BERT training
    
    Args:
        df: DataFrame containing tokens_normalized and narrative_subnarrative_pairs
        label_mapping: Optional pre-existing label mapping to use
    
    Returns:
        tuple: (texts, labels, label_mapping)
    """
    try:
        # Extract filenames for identification
        filenames = df['filename'].tolist()
        
        # Handle tokens_normalized
        texts = df['tokens_normalized'].tolist()
        texts = [' '.join(tokens) if isinstance(tokens, list) else tokens for tokens in texts]
        
        # Convert narrative_subnarrative_pairs to list if it's a string
        narratives = df['narrative_subnarrative_pairs'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        ).tolist()

        # Create or use label mapping
        if label_mapping is None:
            label_mapping = create_label_mapping(narratives)
            
        # Convert narratives to binary indicator format for multilabel classification
        n_classes = len(label_mapping)
        labels = []
        for narrative_list in narratives:
            label_vector = [0] * n_classes
            for narrative in narrative_list:
                narrative_str = str(narrative)  # Convert narrative dict to string
                if narrative_str in label_mapping:
                    label_vector[label_mapping[narrative_str]] = 1
                else:
                    raise ValueError(f"Unknown narrative: {narrative_str}")
            labels.append(label_vector)

        #logger.info(f"Number of unique labels in mapping: {len(label_mapping)}")
        #logger.info(f"Sample text: {texts[0][:100]}")
        #logger.info(f"Sample label: {labels[0]}")
        
        return texts, labels, label_mapping, filenames

    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        logger.error(f"Sample narrative_subnarrative_pairs: {df['narrative_subnarrative_pairs'].iloc[0]}")
        raise

# Map one hot encoded vectors to label dictionaries for easier qualitative analysis
def map_one_hot_to_labels(one_hot_vectors, reverse_label_mapping):
    return [[[reverse_label_mapping[idx], idx] for idx, value in enumerate(vector) if value == 1] for vector in one_hot_vectors]

def bag_of_words(data):
    """
    Convert list of tokenized sentences to bag of words format

    Args:
        data: List of lists of tokens

    Returns:
        list: Bag of words representation of data
    """
    unique_words = set()
    for sentence in data:
        for token in sentence:
            unique_words.add(token)
    
    unique_words = list(unique_words)
    word_to_index = {word: i for i, word in enumerate(unique_words)}

    bag_of_words = []
    for sentence in data:
        word_freq = [0] * len(unique_words)
        for token in sentence:
            word_freq[word_to_index[token]] += 1
        bag_of_words.append(word_freq)

    return bag_of_words