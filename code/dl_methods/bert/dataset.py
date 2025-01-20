import torch
import logging
from typing import List, Dict, Tuple, Set, Optional

logger = logging.getLogger(__name__)

class CustomDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading BERT input data"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def create_label_mapping(all_narratives: List[List[Dict]]) -> Dict[str, int]:
    """
    Create a consistent mapping for all narrative pairs
    """
    unique_narratives: Set[str] = set()
    for narratives in all_narratives:
        for narrative in narratives:
            narrative_str = str(narrative)
            unique_narratives.add(narrative_str)
    
    narrative_to_idx = {
        narrative: idx 
        for idx, narrative in enumerate(sorted(unique_narratives))
    }
    
    logger.info(f"Created mapping for {len(narrative_to_idx)} unique narratives")
    return narrative_to_idx

def get_first_narrative_label(narrative_list: List[Dict], 
                            label_mapping: Dict[str, int]) -> Optional[int]:
    """
    Convert first narrative in list to numeric label
    """
    if narrative_list and len(narrative_list) > 0:
        narrative_str = str(narrative_list[0])
        return label_mapping[narrative_str]
    return None

def prepare_data(df, label_mapping=None) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Prepare data for BERT training
    """
    try:
        texts = df['tokens_normalized'].tolist()
        texts = [' '.join(tokens) if isinstance(tokens, list) else tokens for tokens in texts]
        
        narratives = df['narrative_subnarrative_pairs'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        ).tolist()

        if label_mapping is None:
            label_mapping = create_label_mapping(narratives)
            
        labels = []
        for narrative_list in narratives:
            if narrative_list:
                label_str = str(narrative_list[0])
                if label_str in label_mapping:
                    labels.append(label_mapping[label_str])
                else:
                    raise ValueError(f"Unknown narrative: {label_str}")
            else:
                raise ValueError("Empty narrative list found")

        logger.info(f"Number of unique labels in mapping: {len(label_mapping)}")
        logger.info(f"Sample text: {texts[0][:100]}")
        logger.info(f"Sample label: {labels[0]}")
        
        return texts, labels, label_mapping

    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        logger.error(f"Sample narrative_subnarrative_pairs: {df['narrative_subnarrative_pairs'].iloc[0]}")
        raise