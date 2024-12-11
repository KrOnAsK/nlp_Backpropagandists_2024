from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run_non_dl_method(df, method, representation, split_ratio=0.8, seed=42):
    # X
    x = None
    if representation == "bag_of_words":
        x = bag_of_words(df["tokens_normalized"])
    elif representation == "tf_idf":
        x = tf_idf(df["tokens_normalized"])
    elif representation == "word_embeddings":
        x = word_embeddings(df["content"])
    else:
        raise ValueError(f"Invalid representation: {representation}")

    # Y
    y =  df["topic"].values

    # Run method
    if method == "naive_bayes":
        return naive_bayes(x, y, split_ratio, seed)
    else:
        raise ValueError(f"Invalid method: {method}")
        
    return None


def bag_of_words(data):

    print("Creating bag of words representation...")

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

    print(f"Bag of words representation created with {len(unique_words)} unique words")
    return bag_of_words



def tf_idf(data):
    raise NotImplementedError("TF-IDF representation not implemented yet")


def word_embeddings(data):
    raise NotImplementedError("Word embeddings representation not implemented yet")


def naive_bayes(data, labels, split_ratio=0.8, seed=42):
    """
    Train and evaluate a Naive Bayes classifier
    """

    print("Running Naive Bayes classifier...")

    # Split data
    data, labels = shuffle(data, labels, random_state=seed)
    split_index = int(len(data) * split_ratio)
    train_data, test_data = data[:split_index], data[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]
    

    print(f"Training data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")

    # Train classifier
    clf = MultinomialNB()
    clf.fit(train_data, train_labels)

    # Evaluate classifier
    y_pred = clf.predict(test_data)
    
    accuracy = accuracy_score(test_labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, y_pred, average='weighted')

    print("Naive Bayes classifier completed")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
