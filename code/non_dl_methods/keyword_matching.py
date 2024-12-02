from collections import Counter

def get_word_frequencies_per_narrative(filepath, top_n=20):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    sentences = content.strip().split('\n\n')
    word_frequencies = {}
    current_narrative = []
    
    # Group sentences by narrative (look for text changes)
    prev_text = None
    narratives = {}
    current_sentences = []
    
    for sentence in sentences:
        lines = sentence.split('\n')
        text_line = next((line for line in lines if line.startswith('# text =')), None)
        if text_line:
            text = text_line.replace('# text = ', '')
            if text != prev_text:
                if prev_text:
                    narratives[prev_text] = '\n\n'.join(current_sentences)
                current_sentences = []
                prev_text = text
            current_sentences.append(sentence)
    
    # Add last narrative
    if prev_text:
        narratives[prev_text] = '\n\n'.join(current_sentences)
    
    # Process each narrative
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'is', 'are', 'was', 'were', 'will', 'be', 'have', 'has', 'had','say'}
    
    for text, narrative in narratives.items():
        words = text.lower().split()
        clean_words = [w for w in words if w not in stop_words and w.isalpha() and len(w) > 2]
        counter = Counter(clean_words)
        word_frequencies[text[:50] + "..."] = counter.most_common(top_n)
    
    return word_frequencies

# Usage
filepath = "train.conllu"
frequencies = get_word_frequencies_per_narrative(filepath)

for narrative, words in frequencies.items():
    print(f"\nNarrative: {narrative}")
    print("Most common words:")
    for word, count in words:
        print(f"{word}: {count}")