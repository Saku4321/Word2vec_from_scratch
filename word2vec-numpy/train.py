from data import load_text, build_vocab, text_to_ids
from pairs import generate_pairs, get_negative_samples
from model import Word2Vec
import numpy as np

EMBED_DIM = 100
WINDOW = 2
NEG_SAMPLES = 5
LR = 0.025
EPOCHS = 100

words = load_text('corpus.txt')
word2idx, idx2word = build_vocab(words,min_count=2)
ids = text_to_ids(words,word2idx)
pairs = generate_pairs(ids,window_size=WINDOW)

model = Word2Vec(vocab_size=len(word2idx), embedding_dim=EMBED_DIM)

for epoch in range(EPOCHS):
    total_loss = 0
    np.random.shuffle(pairs)
    
    for center_id,pos_id in pairs:
        neg_ids = get_negative_samples(len(word2idx),pos_id,k=NEG_SAMPLES)

        loss,grad_center_vec,grad_pos_vec,grad_neg_vec = model.forward_and_loss(center_id,pos_id,neg_ids)
        model.update(center_id,pos_id,neg_ids,grad_center_vec,grad_pos_vec,grad_neg_vec,lr=LR)
        total_loss += loss

    print(f"Epoch {epoch+1}/ {EPOCHS} Loss: {total_loss/len(pairs):.4f}")

np.save('W_in.npy',model.W_in)
np.save('word2idx.npy',word2idx)
np.save('idx2word.npy',idx2word)