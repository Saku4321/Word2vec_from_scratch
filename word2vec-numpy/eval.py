import numpy as np

W_in = np.load('W_in.npy')
word2idx = np.load('word2idx.npy', allow_pickle=True).item()
idx2word = np.load('idx2word.npy', allow_pickle=True).item()



def most_similar(word,topn=5):
    if word not in word2idx:
        print(f"'{word}' not in word2idx")
        return []
    vec = W_in[word2idx[word]]

    norms = np.linalg.norm(W_in,axis=1)
    norm_vec = np.linalg.norm(vec)

    similarities = W_in @ vec / (norms * norm_vec + 1e-10)

    top_ids = np.argsort(similarities)[::-1]
    top_ids = top_ids[1:topn + 1]
    return [(idx2word[i],round(float(similarities[i]),3)) for i in top_ids]

def analogy(pos1, neg1, pos2):
    for word in [pos1, neg1, pos2]:
        if word not in word2idx:
            print(f"'{word}' not in word2idx")
            return None
    vec = W_in[word2idx[pos1]] \
    - W_in[word2idx[neg1]] \
    + W_in[word2idx[pos2]]

    norms = np.linalg.norm(W_in,axis=1)
    norm_vec = np.linalg.norm(vec)
    similarities = W_in @ vec / (norms * norm_vec + 1e-10)

    for w in [pos1, neg1, pos2]:
        similarities[word2idx[w]] = -1

    best_id = np.argmax(similarities)
    return idx2word[best_id]


print("\n" + "="*50)
print("CLOSEST WORDS")
print("="*50)

for word in ["cat", "dog", "animal", "forest", "ocean"]:
    similar = most_similar(word, topn=5)
    if similar:
        print(f"Most similar to '{word}'")
        for w, score in similar:
            print(f" {w}: {score}")

print("\n" + "="*50)
print("ANALOGIES")
print("="*50)

test = [
    ("cat", "animal", "dog"),
    ("ocean", "water", "forest"),
    ("lion", "africa", "tiger")
]
for pos1, neg1, pos2 in test:
    result = analogy(pos1, neg1, pos2)
    if result:
        print(f"{pos1} - {neg1} + {pos2} = {result}")
