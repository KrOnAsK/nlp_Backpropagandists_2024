from collections import Counter

def get_top_words_per_narrative(filepath, top_n=5):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    sentences = content.strip().split('\n\n')
    narratives = {}
    current_sentences = []
    prev_text = None
    
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
    
    if prev_text:
        narratives[prev_text] = '\n\n'.join(current_sentences)
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'is', 'are', 'was', 'were', 'will', 'be', 'have', 'has', 'had'}
    
    narrative_words = {}
    for text in narratives:
        words = text.lower().split()
        clean_words = [w for w in words if w not in stop_words and w.isalpha() and len(w) > 2]
        counter = Counter(clean_words)
        narrative_words[text] = [word for word, _ in counter.most_common(top_n)]
    
    return narrative_words

def match_narrative(test_text, train_words, min_matches=2):
    test_words = test_text.lower().split()
    test_words = [w for w in test_words if w.isalpha() and len(w) > 2]
    
    best_match = None
    max_matches = 0
    
    for train_text, top_words in train_words.items():
        matches = sum(1 for word in test_words if word in top_words)
        if matches > max_matches and matches >= min_matches:
            max_matches = matches
            best_match = train_text
    
    return best_match, max_matches

# Usage
train_file = "train.conllu"
test_file = "test.conllu"

# Get top words from training narratives
train_words = get_top_words_per_narrative(train_file)

# Process test file
with open(test_file, 'r', encoding='utf-8') as file:
    test_content = file.read()

test_sentences = test_content.strip().split('\n\n')
results = []

for sentence in test_sentences:
    lines = sentence.split('\n')
    text_line = next((line for line in lines if line.startswith('# text =')), None)
    if text_line:
        test_text = text_line.replace('# text = ', '')
        matched_narrative, num_matches = match_narrative(test_text, train_words)
        if matched_narrative:
            results.append((test_text[:50] + "...", matched_narrative[:50] + "...", num_matches))

print("\nMatching Results:")
for test, match, count in results:
    print(f"\nTest narrative: {test}")
    print(f"Matched with: {match}")
    print(f"Number of matching words: {count}")