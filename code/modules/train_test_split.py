import os
import random

def combine_and_split_conllu(input_dir, train_file, test_file, test_size=0.3, random_seed=42):
    # Collect all sentences from all files
    all_sentences = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.conllu'):
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                sentences = content.strip().split('\n\n')
                all_sentences.extend(sentences)
    
    # Shuffle sentences
    random.seed(random_seed)
    random.shuffle(all_sentences)
    
    # Calculate split point
    split_point = int(len(all_sentences) * (1 - test_size))
    
    # Split into train/test
    train_sentences = all_sentences[:split_point]
    test_sentences = all_sentences[split_point:]
    
    # Write files
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_sentences))
        f.write('\n\n')
        
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(test_sentences))
        f.write('\n\n')
    
    return len(train_sentences), len(test_sentences)

# Usage
input_dir = "./CoNLL"
train_file = "train.conllu"
test_file = "test.conllu"

train_size, test_size = combine_and_split_conllu(input_dir, train_file, test_file)
print(f"Split complete: {train_size} train sentences, {test_size} test sentences")