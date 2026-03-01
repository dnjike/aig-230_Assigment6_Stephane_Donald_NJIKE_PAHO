# NLP Language Modeling Project  
**Next-Token Prediction with N-gram and RNN Models**

## 📌 Overview

This project implements an end-to-end Natural Language Processing (NLP) pipeline using the NLTK Brown corpus (news category).  

The objective is to:

- Preprocess text data
- Create vector representations
- Train an n-gram statistical language model
- Train a Recurrent Neural Network (RNN) language model
- Evaluate models using perplexity
- Generate text samples

This project demonstrates understanding of both classical and neural language modeling techniques.

---

## 📂 Dataset

**Source:** NLTK Brown Corpus  
**Category Used:** `news`

The dataset was:
- Sentence-segmented
- Tokenized
- Converted into a continuous token stream
- Split into train / validation / test sets

Special tokens used:
- `<bos>` – beginning of sentence  
- `<eos>` – end of sentence  
- `<unk>` – unknown token  

---

## 🧠 Models Implemented

### 1️⃣ N-gram Language Model
- Implemented with add-k smoothing
- Trained on training split
- Evaluated using perplexity
- Serves as statistical baseline

---

### 2️⃣ RNN Language Model (PyTorch)

Architecture:

- Embedding layer
- Vanilla RNN
- Linear output layer
- CrossEntropyLoss
- Adam optimizer

Forward Pass:

- Input → Embedding
- Embedding → RNN
- RNN output → Linear layer
- Output logits returned

Training details:

- Sequence length: configurable
- Batch training using DataLoader
- Gradient clipping applied
- Perplexity used for evaluation

---

## 📊 Training Procedure

For each epoch:

1. Zero gradients  
2. Forward pass  
3. Reshape logits to `(B*T, V)`  
4. Reshape targets to `(B*T,)`  
5. Compute CrossEntropyLoss  
6. Backpropagation  
7. Gradient clipping  
8. Optimizer step  

Metrics reported per epoch:

- Training loss  
- Validation perplexity  

A training loss curve is plotted.

---

## 📈 Evaluation

Final evaluation includes:

- Validation perplexity
- Test perplexity
- Comparison between n-gram and RNN models

### Interpretation

- Lower perplexity indicates better predictive performance.
- RNN model captures longer context dependencies compared to n-gram.
- Neural model typically achieves lower perplexity due to distributed representations.

---

## ✍️ Text Generation

Text is generated using:

- `<bos>` token as starting point
- Autoregressive next-token sampling
- Softmax + multinomial sampling
- Stops at `<eos>` or max length

Analysis includes:

- Grammaticality
- Coherence
- Repetition behavior
- Long-range dependency behavior

---

## 📦 Requirements

- Python 3.x
- PyTorch
- NLTK
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install torch nltk numpy matplotlib