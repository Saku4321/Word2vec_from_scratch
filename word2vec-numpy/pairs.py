import numpy as np
def generate_pairs(ids, window_size=2):
    pairs = []
    for i,center in enumerate(ids):
        left = max(0, i- window_size)
        right = min(len(ids),i+ window_size +1)
        for j in range(left,right):
            if i==j:
                continue
            pairs.append((center,ids[j]))

    return pairs

def get_negative_samples(vocab_size, exclude, k=5):
    negs = []
    while len(negs) < k:
        sample = np.random.randint(0,vocab_size)

        if sample != exclude:
            negs.append(sample)
    return negs


