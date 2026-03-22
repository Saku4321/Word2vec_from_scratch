# Word2Vec from Scratch in NumPy

Implementation of the Word2Vec Skip-gram model with Negative Sampling, written in pure NumPy — no PyTorch, TensorFlow, or any other ML framework.

---

## Overview

This project implements the core Word2Vec training loop from scratch, including:

- Skip-gram architecture
- Negative sampling loss (binary cross-entropy)
- Forward pass, gradient derivation, and SGD parameter updates
- Cosine similarity evaluation and word analogy testing

---

## How It Works

Skip-gram takes a center word and tries to predict its surrounding context words within a fixed window. Instead of computing softmax over the entire vocabulary (which is expensive), **negative sampling** is used: for each real (center, context) pair, `k` random words are sampled as negatives. The model learns to assign high probability to real pairs and low probability to random ones.

The loss function:

```
L = -log σ(v_c · v_pos) - Σ log σ(-v_c · v_neg_k)
```

where `σ` is the sigmoid function, `v_c` is the center word vector, `v_pos` is the true context vector, and `v_neg` are the negative sample vectors.

Gradients:

```
∂L/∂v_c   = (σ(v_c · v_pos) - 1) · v_pos  +  Σ σ(v_c · v_neg) · v_neg
∂L/∂v_pos = (σ(v_c · v_pos) - 1) · v_c
∂L/∂v_neg = σ(v_c · v_neg) · v_c
```

Parameters are updated with standard SGD:
```
W -= lr * gradient
```

---

## Project Structure

```
word2vec-numpy/
├── data.py       # load text, build vocabulary, convert words to indices
├── pairs.py      # generate (center, context) pairs + negative sampling
├── model.py      # W_in, W_out matrices, forward pass, loss, gradients, update
├── train.py      # training loop, saves W_in.npy to disk
├── eval.py       # most_similar() and analogy() evaluation
└── corpus.txt    # training corpus
```

---

## Quickstart

```bash
git clone https://github.com/Saku4321/Word2vec_from_scratch.git
cd Word2vec_from_scratch
pip install numpy
python train.py
python eval.py
```

---

## Training

Key hyperparameters in `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBED_DIM` | 100 | Dimensionality of word vectors |
| `WINDOW` | 2 | Context window size |
| `NEG_SAMPLES` | 5 | Number of negative samples per pair |
| `LR` | 0.025 | Learning rate |
| `EPOCHS` | 100 | Number of passes over the data |

Training output:
```
Epoch 1/100  Loss: 4.1578
Epoch 2/100  Loss: 4.1366
Epoch 3/100  Loss: 3.8685
...
Epoch 100/100  Loss: 1.9823
```

Loss decreases each epoch — embeddings are learning.

---

## Evaluation

After training, `eval.py` loads the saved embeddings and runs two checks:

**Most similar words** (cosine similarity):
```
Most similar to 'cat':
  mouse: 0.821
  pets:  0.651
  dog:   0.623

Most similar to 'ocean':
  dolphins: 0.728
  fish:     0.705
  whales:   0.683
```

**Word analogies** (`pos1 - neg1 + pos2 = ?`):
```
cat - animal + dog = mouse
ocean - water + forest = dolphins
```

---

## Dataset

The included `corpus.txt` is a small custom corpus (~500 words) focused on animals and nature — sufficient to demonstrate that semantically related words cluster together in the embedding space.

For stronger results, replace `corpus.txt` with a larger corpus such as [text8](http://mattmahoney.net/dc/textdata.html) and increase `EPOCHS`.

---

## Dependencies

```
numpy
```

Install with:
```bash
pip install numpy
```
