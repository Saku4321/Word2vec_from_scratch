import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=100):
        self.W_in = np.random.randn(vocab_size, embedding_dim)*0.01
        self.W_out = np.random.randn(vocab_size, embedding_dim)*0.01
        self.embedding_dim = embedding_dim

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def forward_and_loss(self, center_id, pos_id, neg_ids):
        center_vec = self.W_in[center_id]
        pos_vec=self.W_out[pos_id]
        negs_vec=self.W_out[neg_ids]

        pos_score=center_vec @ pos_vec
        neg_scores = negs_vec @ center_vec

        sig_pos = self.sigmoid(pos_score)
        sig_negs = self.sigmoid(neg_scores)

        loss = (-np.log(sig_pos+ 1e-10) \
                - np.sum(np.log(1 - sig_negs + 1e-10)))

        err_pos = sig_pos - 1.0
        err_negs = sig_negs

        grad_center_vec = err_pos * pos_vec + err_negs @ negs_vec
        grad_pos_vec = err_pos * center_vec
        grad_negs_vec = np.outer(err_negs,center_vec)

        return loss,grad_center_vec,grad_pos_vec,grad_negs_vec

    def update(self, center_id, pos_id, neg_ids,
               grad_center_vec, grad_pos_vec, grad_negs_vec, lr=0.025):
        self.W_in[center_id] -= lr * grad_center_vec
        self.W_out[pos_id] -= lr * grad_pos_vec
        self.W_out[neg_ids] -= lr * grad_negs_vec
