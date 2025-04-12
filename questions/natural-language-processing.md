# Natural Language Processing Questions

This file contains natural language processing (NLP) questions commonly asked in machine learning interviews at companies like **Google**, **Amazon**, **Meta**, and others. These questions assess your **understanding** of NLP techniques, models, and applications, covering topics like text preprocessing, embeddings, transformers, and evaluation.

Below are the questions with detailed answers, including explanations, technical details, and practical insights for interviews.

---

## Table of Contents

1. [What are the key steps in preprocessing text data for NLP tasks?](#1-what-are-the-key-steps-in-preprocessing-text-data-for-nlp-tasks)
2. [What is tokenization, and why is it important in NLP?](#2-what-is-tokenization-and-why-is-it-important-in-nlp)
3. [Explain the difference between bag-of-words and TF-IDF representations](#3-explain-the-difference-between-bag-of-words-and-tf-idf-representations)
4. [What are word embeddings, and how do they improve NLP models?](#4-what-are-word-embeddings-and-how-do-they-improve-nlp-models)
5. [What is the difference between RNNs and transformers for NLP tasks?](#5-what-is-the-difference-between-rnns-and-transformers-for-nlp-tasks)
6. [Explain the attention mechanism in the context of NLP](#6-explain-the-attention-mechanism-in-the-context-of-nlp)
7. [What is BERT, and how does it differ from traditional word embeddings?](#7-what-is-bert-and-how-does-it-differ-from-traditional-word-embeddings)
8. [How do you evaluate the performance of an NLP model?](#8-how-do-you-evaluate-the-performance-of-an-nlp-model)

---

## 1. What are the key steps in preprocessing text data for NLP tasks?

**Answer**:

Preprocessing text data transforms raw text into a format suitable for NLP models. Key steps:

- **Lowercasing**:
  - Convert text to lowercase (e.g., “Hello” → “hello”).
  - **Why**: Reduces vocabulary size, ensures consistency.
- **Tokenization**:
  - Split text into tokens (words, subwords, or characters).
  - **Why**: Enables numerical representation.
  - **Tool**: NLTK, spaCy, or model-specific (e.g., BERT tokenizer).
- **Removing Noise**:
  - Strip punctuation, special characters, or URLs.
  - **Why**: Reduces irrelevant features.
- **Stop Word Removal**:
  - Remove common words (e.g., “the,” “is”).
  - **Why**: Focuses on meaningful terms, but context-dependent (e.g., keep for transformers).
- **Stemming/Lemmatization**:
  - Reduce words to root form (e.g., “running” → “run”).
  - **Why**: Normalizes variations, but lemmatization is context-aware.
  - **Tool**: Porter Stemmer, WordNet lemmatizer.
- **Handling Numbers**:
  - Normalize or remove numbers (e.g., “2023” → “<NUMBER>”).
  - **Why**: Generalizes numerical data.
- **N-grams** (optional):
  - Extract multi-word sequences (e.g., “machine learning”).
  - **Why**: Captures phrases for some tasks.
- **Encoding**:
  - Convert tokens to IDs for model input (e.g., vocabulary indices).
  - **Why**: Models require numerical input.

**Example**:
- Raw: “I’m running to the Store in 2023!”
- Processed: Tokens = [“run”, “store”] (after lowercasing, removing noise, lemmatizing).

**Interview Tips**:
- Tailor steps: “Depends on task—transformers need less cleanup.”
- Mention tools: “spaCy for pipelines, BERT for tokenization.”
- Be ready to code: “Show tokenization in Python.”

---

## 2. What is tokenization, and why is it important in NLP?

**Answer**:

**Tokenization** is the process of splitting text into smaller units (tokens), such as words, subwords, or characters, to enable numerical processing by NLP models.

- **Types**:
  - **Word**: Split on spaces/punctuation (e.g., “I love NLP” → [“I”, “love”, “NLP”]).
  - **Subword**: Break words into pieces (e.g., “playing” → [“play”, “##ing”]).
    - Used in BERT, WordPiece, BPE.
  - **Character**: Split into individual characters (e.g., “NLP” → [“N”, “L”, “P”]).
- **Why Important**:
  - **Numerical Input**: Models require tokens to map to IDs (e.g., vocabulary).
  - **Granularity**: Captures meaning at right level (e.g., subwords handle rare words).
  - **Context Preservation**: Maintains structure for tasks like translation.
  - **Vocabulary Size**: Balances size vs. coverage (subwords reduce OOV).
- **Challenges**:
  - Ambiguity (e.g., “U.S.” vs. “us”).
  - Language-specific rules (e.g., Chinese segmentation).
  - Reversible tokenization for generation.

**Example**:
- Text: “unhappiness”.
- Subword: [“un”, “##happiness”] (BERT-style).
- Benefit: Handles “happy” and “unhappy” consistently.

**Interview Tips**:
- Explain types: “Subword is key for modern NLP.”
- Link to models: “BERT uses WordPiece for flexibility.”
- Be ready to sketch: “Show text → tokens → IDs.”

---

## 3. Explain the difference between bag-of-words and TF-IDF representations

**Answer**:

- **Bag-of-Words (BoW)**:
  - **How**: Represent text as a vector of word counts or presence (binary).
  - **Process**:
    - Build vocabulary (e.g., all unique words).
    - Vectorize: Count occurrences per document (e.g., [2, 0, 1] for words).
  - **Pros**: Simple, captures word frequency.
  - **Cons**: Ignores word order, no semantic meaning, high-dimensional.
  - **Use Case**: Basic text classification (e.g., spam detection).
  - **Example**: “cat dog” → [1, 1, 0] (vocab: cat, dog, bird).

- **TF-IDF (Term Frequency-Inverse Document Frequency)**:
  - **How**: Weight words by frequency in document (TF) and rarity across corpus (IDF).
  - **Formula**:
    - TF = `count(word, doc) / len(doc)`.
    - IDF = `log(N / n_t)`, where `N` is docs, `n_t` is docs with word.
    - TF-IDF = `TF * IDF`.
  - **Pros**: Downweights common words (e.g., “the”), highlights distinctive terms.
  - **Cons**: Still ignores order, sparse vectors.
  - **Use Case**: Document retrieval, topic modeling.
  - **Example**: “cat” in one doc, rare in corpus → high TF-IDF.

- **Key Differences**:
  - **Weighting**: BoW uses raw counts; TF-IDF adjusts for rarity.
  - **Meaning**: BoW treats all words equally; TF-IDF emphasizes uniqueness.
  - **Sparsity**: Both sparse, but TF-IDF reduces noise from frequent words.
  - **ML Context**: BoW for simple models; TF-IDF for search/relevance.

**Example**:
- Docs: “cat dog”, “dog bird”.
- BoW: [1, 1, 0], [0, 1, 1].
- TF-IDF: “cat” gets higher weight (rare), “dog” lower (common).

**Interview Tips**:
- Clarify limits: “Both lose context, unlike embeddings.”
- Explain IDF: “Rarity boosts important words.”
- Be ready to compute: “Show TF-IDF for a small corpus.”

---

## 4. What are word embeddings, and how do they improve NLP models?

**Answer**:

**Word embeddings** are dense, low-dimensional vector representations of words that capture semantic meaning, learned from text data.

- **How They Work**:
  - Map words to vectors (e.g., 300-dim) in a continuous space.
  - Similar words (e.g., “king”, “queen”) are close in vector space.
  - Learned via models like:
    - **Word2Vec**: Predict word given context (CBOW) or vice versa (Skip-gram).
    - **GloVe**: Factorize co-occurrence matrix.
    - **FastText**: Include subword info for rare words.

- **Why They Improve NLP**:
  - **Semantic Similarity**: Capture relationships (e.g., “king - man + woman ≈ queen”).
  - **Dimensionality Reduction**: Dense (vs. sparse BoW), reduces parameters.
  - **Generalization**: Handle unseen words via similarity (e.g., FastText subwords).
  - **Transfer Learning**: Pretrained embeddings (e.g., GloVe) boost small datasets.
  - **Contextual Models**: Lead to advanced embeddings (e.g., BERT, contextual).

- **Limitations**:
  - Static embeddings lack context (e.g., “bank” as river vs. finance).
  - Compute-intensive to train from scratch.

**Example**:
- Task: Sentiment analysis.
- GloVe: “happy” = [0.1, 0.5, ...], “joy” nearby → model learns positive sentiment.

**Interview Tips**:
- Highlight semantics: “Embeddings encode meaning, not just counts.”
- Compare: “BoW is sparse, embeddings are dense.”
- Be ready to sketch: “Show ‘king’, ‘queen’ in 2D space.”

---

## 5. What is the difference between RNNs and transformers for NLP tasks?

**Answer**:

- **RNNs (Recurrent Neural Networks)**:
  - **How**: Process sequences sequentially, maintaining a hidden state.
    - Update: `h_t = f(W_h * h_{t-1} + W_x * x_t + b)`.
  - **Variants**: LSTM, GRU mitigate vanishing gradients.
  - **Pros**:
    - Good for short sequences.
    - Memory-efficient for small models.
  - **Cons**:
    - Sequential processing → slow training/inference.
    - Struggles with long-range dependencies (e.g., 100+ tokens).
  - **Use Case**: Early NLP (e.g., sentiment with short texts).
  - **Example**: Predict next word in “I love…” using LSTM.

- **Transformers**:
  - **How**: Process sequences in parallel using self-attention.
    - Attention: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`.
    - Stack encoder/decoder layers for tasks.
  - **Pros**:
    - Captures long-range dependencies (e.g., sentence-wide context).
    - Parallelizable → faster training.
    - Scales to large models (e.g., BERT, GPT).
  - **Cons**:
    - Memory-intensive (quadratic w.r.t. sequence length).
    - Requires large data to train.
  - **Use Case**: Modern NLP (e.g., translation, QA).
  - **Example**: BERT understands “bank” context in sentence.

- **Key Differences**:
  - **Processing**: RNNs are sequential; transformers are parallel.
  - **Dependencies**: RNNs struggle with long-range; transformers excel.
  - **Scalability**: Transformers dominate large-scale NLP; RNNs for niche cases.
  - **ML Context**: Transformers replaced RNNs for most tasks (e.g., BERT vs. LSTM).

**Example**:
- RNN: Fails to link “he” to “John” in long text.
- Transformer: Captures link via attention, better accuracy.

**Interview Tips**:
- Emphasize attention: “Transformers focus on relevant tokens.”
- Discuss speed: “Parallelism makes transformers faster.”
- Be ready to sketch: “Show RNN loop vs. transformer attention.”

---

## 6. Explain the attention mechanism in the context of NLP

**Answer**:

The **attention mechanism** allows NLP models to focus on relevant parts of a sequence when processing or generating text, improving context understanding.

- **How It Works**:
  - **Scaled Dot-Product Attention**:
    - Input: Query (`Q`), Key (`K`), Value (`V`) vectors for each token.
    - Compute: `Attention = softmax(QK^T/√d_k)V`.
    - Output: Weighted sum of values, emphasizing important tokens.
  - **Multi-Head Attention**:
    - Run multiple attention layers in parallel, concatenate outputs.
    - Captures different relationships (e.g., syntax, semantics).
  - **Self-Attention**:
    - Tokens attend to each other in same sequence (e.g., sentence).
  - **Math**:
    - Similarity: `QK^T` measures token relevance.
    - Scaling: `√d_k` stabilizes gradients.

- **In NLP**:
  - **Transformers**: Core of BERT, GPT, enabling context-aware embeddings.
  - **Tasks**:
    - Translation: Focus on source words for target.
    - QA: Attend to relevant sentence parts.
    - Summarization: Highlight key phrases.
  - **Benefits**:
    - Long-range dependencies (unlike RNNs).
    - Interpretable weights (e.g., see what “it” refers to).

**Example**:
- Sentence: “The cat, which is black, sleeps.”
- Attention: “Sleeps” attends to “cat,” not “black,” for meaning.

**Interview Tips**:
- Simplify intuition: “Attention weighs important words.”
- Link to transformers: “Powers modern NLP models.”
- Be ready to derive: “Show attention matrix calculation.”

---

## 7. What is BERT, and how does it differ from traditional word embeddings?

**Answer**:

**BERT** (Bidirectional Encoder Representations from Transformers) is a pretrained transformer model that generates contextual word embeddings for NLP tasks.

- **How BERT Works**:
  - **Architecture**: Stack of transformer encoders (e.g., 12 layers in BERT-base).
  - **Pretraining**:
    - **Masked Language Model (MLM)**: Predict masked words (15% of tokens).
    - **Next Sentence Prediction (NSP)**: Predict if sentences follow each other.
  - **Fine-Tuning**: Adapt to tasks (e.g., classification, QA) with task-specific data.
  - **Output**: Contextual embeddings for each token, varying by sentence.

- **Traditional Word Embeddings**:
  - **Examples**: Word2Vec, GloVe, FastText.
  - **How**: Static vectors per word, trained on co-occurrence or prediction.
  - **Properties**: Same vector for “bank” in all contexts.

- **Key Differences**:
  - **Contextuality**:
    - BERT: Dynamic embeddings (e.g., “bank” differs in “river bank” vs. “bank account”).
    - Traditional: Fixed embeddings, no context.
  - **Directionality**:
    - BERT: Bidirectional, considers full sentence.
    - Traditional: Unidirectional or context-free.
  - **Training**:
    - BERT: Pretrained on large corpus, fine-tuned.
    - Traditional: Trained once, used as-is or lightly adapted.
  - **Performance**:
    - BERT: Superior for tasks like QA, sentiment (context matters).
    - Traditional: Simpler, faster for basic tasks.
  - **ML Context**: BERT powers modern NLP; traditional embeddings for lightweight models.

**Example**:
- Sentence: “I bank money.”
- BERT: “bank” embedding reflects finance.
- GloVe: Same “bank” vector for all uses.

**Interview Tips**:
- Highlight context: “BERT adapts to sentence meaning.”
- Compare size: “BERT is heavy, GloVe is light.”
- Be ready to sketch: “Show BERT’s transformer layers.”

---

## 8. How do you evaluate the performance of an NLP model?

**Answer**:

Evaluating NLP models depends on the task, using metrics to measure accuracy, robustness, and generalization:

- **Classification Tasks** (e.g., sentiment analysis):
  - **Accuracy**: Fraction of correct predictions.
  - **Precision/Recall/F1**:
    - Precision: `TP / (TP + FP)` (correct positives).
    - Recall: `TP / (TP + FN)` (captured positives).
    - F1: `2 * (P * R) / (P + R)` (harmonic mean).
  - **Use**: F1 for imbalanced data (e.g., spam detection).
- **Sequence Labeling** (e.g., NER):
  - **Token-Level F1**: Evaluate per token (e.g., “B-PER”, “I-PER”).
  - **Entity-Level F1**: Match full entities (e.g., “John Doe”).
  - **Use**: Entity F1 for strict matching.
- **Generation Tasks** (e.g., translation, summarization):
  - **BLEU**: Measures n-gram overlap with reference.
  - **ROUGE**: Recall-oriented for summaries (e.g., ROUGE-L for longest sequence).
  - **METEOR**: Considers synonyms, stemming.
  - **Use**: BLEU for translation, ROUGE for summarization.
- **Question Answering**:
  - **Exact Match (EM)**: Full answer match.
  - **F1 Score**: Overlap of answer tokens.
  - **Use**: EM for strict evaluation, F1 for partial credit.
- **Embedding-Based** (e.g., similarity):
  - **Cosine Similarity**: Measure vector alignment.
  - **Correlation**: Compare to human judgments.
  - **Use**: Evaluate semantic search.
- **Human Evaluation**:
  - Subjective scoring for fluency, coherence (e.g., chatbot responses).
  - **Use**: When metrics fail (e.g., creative tasks).
- **General Practices**:
  - **Cross-Validation**: Ensure robustness.
  - **Error Analysis**: Inspect mispredictions (e.g., confusion matrix).
  - **Domain-Specific**: Align metrics with goals (e.g., latency for real-time).

**Example**:
- Task: Sentiment classification.
- Metrics: F1 = 0.85 (handles imbalance), accuracy = 0.88.
- Analysis: Check false positives for improvement.

**Interview Tips**:
- Match metric to task: “F1 for classification, BLEU for translation.”
- Discuss limits: “BLEU misses fluency, needs human eval.”
- Be ready to compute: “Show F1 formula.”

---

## Notes

- **Focus**: Answers cover NLP fundamentals and advanced models, ideal for ML interviews.
- **Clarity**: Explanations are structured for verbal delivery, with examples and trade-offs.
- **Depth**: Includes technical details (e.g., attention math, BERT pretraining) and practical tips (e.g., preprocessing pipelines).
- **Consistency**: Matches the style of previous files for a cohesive repository.

For deeper practice, implement NLP models (see [ML Coding](ml-coding.md)) or explore [Deep Learning](deep-learning.md) for transformer foundations. 🚀

---

**Next Steps**: Build on these skills with [Statistics & Probability](statistics-probability.md) for NLP evaluation metrics or revisit [Production MLOps](production-mlops.md) for deploying NLP models! 🌟