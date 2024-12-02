import os
from collections import Counter

def read_conllu_files(directory):
    all_text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.conllu'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                all_text += file.read() + "\n\n"
    return all_text

def get_most_common_words(conllu_text, top_n=20):
    sentences = conllu_text.strip().split('\n\n')
    narratives = []
    
    for sentence in sentences:
        lines = sentence.split('\n')
        for line in lines:
            if line.startswith('# text ='):
                text = line.replace('# text = ', '')
                narratives.append(text)
                break
    
    word_frequencies = {}
    for narrative in narratives:
        words = narrative.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'is', 'are', 'was', 'were', 'will', 'be', 'have', 'has', 'had'}
        words = [w for w in words if w not in stop_words and w.isalpha() and len(w) > 2]
        counter = Counter(words)
        word_frequencies[narrative[:50] + "..."] = counter.most_common(top_n)
        
    return word_frequencies

# Usage
directory = "./CoNLL"  # Replace with actual path
conllu_text = read_conllu_files(directory)
frequencies = get_most_common_words(conllu_text)

for narrative, words in frequencies.items():
    print(f"\nNarrative: {narrative}")
    print("Most common words:")
    for word, count in words:
        print(f"{word}: {count}")