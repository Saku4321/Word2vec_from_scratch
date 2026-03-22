import numpy as np
def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text =f.read().lower()
    words=text.split()
    return words

def build_vocab(words, min_count=2):
    from collections import Counter
    count = Counter(words)
    vocab = [w for w, c in count.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return word2idx, idx2word

def text_to_ids(words, word2idx):
    return [word2idx[w] for w in words if w in word2idx]

