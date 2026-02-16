# andrew-ng-nlp-study
Curated resources, notes, and implementations from the Coursera NLP Specialization by DeepLearning.AI. Covering Sentiment Analysis to Attention Models.

# NLP Specialization Vault 🧠

A comprehensive reference hub and personal "second brain" for Natural Language Processing. This repository contains curated resources, mathematical foundations, and implementation notes from the **DeepLearning.AI NLP Specialization** (by Andrew Ng and Younes Bensouda Mourri).

---

## 🗺️ Curriculum Roadmap

### 📂 [1. Classification and Vector Spaces](./Course-1)
*Focus: Foundational machine learning for text.*
* **Sentiment Analysis:** Logistic Regression & Naïve Bayes.
* **Vector Spaces:** Word embeddings, Cosine Similarity, and PCA.
* **Key Technique:** Locality Sensitive Hashing (LSH) for approximate nearest neighbors.

### 📂 [2. Probabilistic Models](./Course-2)
*Focus: Statistical dependencies and sequence labeling.*
* **Autocorrect:** Minimum Edit Distance & String Manipulation.
* **Autocomplete:** N-gram Language Models and Perplexity.
* **POS Tagging:** Hidden Markov Models (HMM) and the Viterbi Algorithm.

### 📂 [3. Sequence Models](./Course-3)
*Focus: Deep learning for temporal data.*
* **Neural Networks:** RNNs, GRUs, and LSTMs for sentiment and NER.
* **Siamese Networks:** Question duplication and similarity learning.
* **Key Concept:** Overcoming Vanishing Gradients in long sequences.

### 📂 [4. Attention Models](./Course-4)
*Focus: Modern SOTA architectures.*
* **Machine Translation:** Encoder-Decoder architectures.
* **Transformers:** Scaled Dot-Product Attention & Multi-head Attention.
* **Advanced Models:** Summarization with T5 and efficient processing with Reformers.

---

## 🛠️ Tech Stack & Tools
* **Library:** [Trax](https://github.com/google/trax) (High-performance DL by Google Brain)
* **Preprocessing:** NLTK, Tokenization, Stemming, Stop-word removal.
* **Math:** NumPy (Linear Algebra), Scikit-learn (Baselines).

---

## 📝 Quick Reference Table

| Concept | Model/Algo | Use Case |
| :--- | :--- | :--- |
| **Similarity** | Cosine Similarity | Document/Word clustering |
| **Sequence** | Viterbi / HMM | Part-of-Speech Tagging |
| **Memory** | LSTMs / GRUs | Long-form text analysis |
| **Context** | Self-Attention | Machine Translation / Transformers |

---

## 🚀 How to Use This Reference
This repository is organized by course and week. For a quick refresher on **Attention Mechanisms**, navigate to `Course-4/`. For **Word Embeddings**, see `Course-1/`.

> *"The goal of this vault is to bridge the gap between theoretical math and production-ready NLP code."*

---
