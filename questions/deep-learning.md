# Deep Learning Questions

This file contains deep learning questions commonly asked in interviews at companies like **Google**, **Meta**, **Amazon**, and others. These questions assess your **in-depth understanding** of neural networks, including architectures, optimization, regularization, and practical applications like computer vision and NLP. They test your ability to articulate complex concepts and design solutions.

Below are the questions with detailed answers, including explanations, mathematical intuition where relevant, and practical insights for interviews.

---

## Table of Contents

1. [What is a neural network?](#1-what-is-a-neural-network)
2. [What is backpropagation and how does it work?](#2-what-is-backpropagation-and-how-does-it-work)
3. [What is the role of activation functions in neural networks?](#3-what-is-the-role-of-activation-functions-in-neural-networks)
4. [What is the vanishing gradient problem and how can it be mitigated?](#4-what-is-the-vanishing-gradient-problem-and-how-can-it-be-mitigated)
5. [What is the difference between batch normalization and layer normalization?](#5-what-is-the-difference-between-batch-normalization-and-layer-normalization)
6. [What are convolutional neural networks (CNNs) and when are they used?](#6-what-are-convolutional-neural-networks-cnns-and-when-are-they-used)
7. [Explain the concept of recurrent neural networks (RNNs)](#7-explain-the-concept-of-recurrent-neural-networks-rnns)
8. [What are LSTMs and how do they address the limitations of RNNs?](#8-what-are-lstms-and-how-do-they-address-the-limitations-of-rnns)
9. [What is the difference between a generative and discriminative model?](#9-what-is-the-difference-between-a-generative-and-discriminative-model)
10. [Explain the concept of transfer learning in deep learning](#10-explain-the-concept-of-transfer-learning-in-deep-learning)
11. [What are attention mechanisms and how are they used in transformers?](#11-what-are-attention-mechanisms-and-how-are-they-used-in-transformers)
12. [What is the difference between SGD, Adam, and RMSprop optimizers?](#12-what-is-the-difference-between-sgd-adam-and-rmsprop-optimizers)
13. [What are GANs and how do they work?](#13-what-are-gans-and-how-do-they-work)
14. [What are autoencoders and what are they used for?](#14-what-are-autoencoders-and-what-are-they-used-for)
15. [How do you prevent overfitting in deep neural networks?](#15-how-do-you-prevent-overfitting-in-deep-neural-networks)
16. [What is the role of pooling layers in CNNs?](#16-what-is-the-role-of-pooling-layers-in-cnns)
17. [What is the difference between 1x1 convolution and regular convolution?](#17-what-is-the-difference-between-1x1-convolution-and-regular-convolution)
18. [What are residual connections and why are they important in deep networks?](#18-what-are-residual-connections-and-why-are-they-important-in-deep-networks)
19. [What is the difference between word2vec and GloVe embeddings?](#19-what-is-the-difference-between-word2vec-and-glove-embeddings)
20. [What is the role of positional encoding in transformers?](#20-what-is-the-role-of-positional-encoding-in-transformers)

---

## 1. What is a neural network?

**Answer**:

A **neural network** is a computational model inspired by the human brain, used to learn complex patterns from data. It consists of interconnected nodes (neurons) organized in layers.

- **Structure**:
  - **Input Layer**: Receives features (e.g., pixel values).
  - **Hidden Layers**: Process data through weighted connections and activation functions.
  - **Output Layer**: Produces predictions (e.g., class probabilities).
- **Operation**:
  - Each neuron computes a weighted sum of inputs, adds a bias, and applies an activation function (e.g., ReLU).
  - Layers transform data hierarchically to capture patterns.
- **Training**:
  - Minimize a loss function (e.g., MSE, cross-entropy) using backpropagation and optimization (e.g., gradient descent).
- **Use Cases**: Image classification, NLP, regression.

**Example**:
- Task: Classify cats vs. dogs.
- Network: Input (image pixels) → Hidden layers (learn edges, shapes) → Output (cat/dog probability).

**Interview Tips**:
- Use analogies: “Like a brain with layers learning patterns.”
- Mention flexibility: “Can model non-linear relationships.”
- Be ready to sketch: “Input → Hidden → Output.”

---

## 2. What is backpropagation and how does it work?

**Answer**:

**Backpropagation** is an algorithm to train neural networks by computing gradients of the loss function with respect to weights, enabling optimization via gradient descent.

- **How It Works**:
  1. **Forward Pass**:
     - Compute predictions by passing inputs through layers.
     - Calculate loss (e.g., cross-entropy).
  2. **Backward Pass**:
     - Compute gradient of loss w.r.t. each weight using the chain rule.
     - Start from output layer, propagate errors backward.
  3. **Update Weights**:
     - Adjust weights using gradients and learning rate (e.g., `w = w - lr * gradient`).
  4. **Repeat**: Iterate until convergence.

- **Math**:
  - For weight `w`, gradient = `∂L/∂w = ∂L/∂z * ∂z/∂w`, where `z` is neuron output.
  - Chain rule computes partial derivatives layer by layer.

**Example**:
- Network: 2-layer NN for classification.
- Forward: Predict probability, compute loss.
- Backward: Adjust weights to reduce errors.

**Interview Tips**:
- Emphasize chain rule: “Backprop computes gradients efficiently.”
- Mention computation: “It’s automatic in frameworks like PyTorch.”
- Be ready to derive: “Show gradient for a single weight.”

---

## 3. What is the role of activation functions in neural networks?

**Answer**:

**Activation functions** introduce non-linearity to neural networks, enabling them to model complex patterns.

- **Role**:
  - Transform weighted inputs (`z = w * x + b`) to outputs.
  - Allow stacking layers to learn hierarchical features (e.g., edges → objects).
  - Prevent network from collapsing to a linear model.

- **Common Functions**:
  - **Sigmoid**: `1/(1 + e^-z)` (0 to 1, for probabilities).
  - **ReLU**: `max(0, z)` (fast, avoids vanishing gradients).
  - **Tanh**: `(e^z - e^-z)/(e^z + e^-z)` (-1 to 1, zero-centered).
  - **Softmax**: Normalizes outputs to probabilities (for multi-class).

**Example**:
- Linear model: `y = w * x` (can’t model XOR).
- With ReLU: Captures non-linear patterns (e.g., image edges).

**Interview Tips**:
- Explain non-linearity: “Without it, layers just stack linearly.”
- Discuss trade-offs: “ReLU is fast but has dead neurons.”
- Mention use case: “Softmax for classification outputs.”

---

## 4. What is the vanishing gradient problem and how can it be mitigated?

**Answer**:

The **vanishing gradient problem** occurs when gradients become very small during backpropagation, slowing or stopping learning in deep networks.

- **Cause**:
  - In deep networks, gradients are multiplied layer by layer (chain rule).
  - Activation functions like sigmoid/tanh squash gradients (e.g., derivative < 1).
  - Early layers receive tiny updates, learning stalls.

- **Mitigation**:
  - **ReLU Activation**: Non-saturating derivative (1 for `z > 0`).
  - **Weight Initialization**: Use Xavier/He initialization to balance variance.
  - **Batch Normalization**: Normalizes layer outputs, stabilizes gradients.
  - **Residual Connections**: Add shortcuts (e.g., ResNet) to ease gradient flow.
  - **LSTM/GRU**: For RNNs, gates maintain gradient flow.
  - **Gradient Clipping**: Cap gradients to prevent explosion.

**Example**:
- Deep RNN with sigmoid: Early layers don’t learn.
- Fix: Use ReLU and batch norm, gradients propagate better.

**Interview Tips**:
- Relate to depth: “Deep networks amplify the issue.”
- Explain fixes: “ReLU keeps gradients alive.”
- Be ready to sketch: “Show gradient shrinking backward.”

---

## 5. What is the difference between batch normalization and layer normalization?

**Answer**:

- **Batch Normalization (BN)**:
  - **How**: Normalizes activations across a mini-batch for each feature.
  - **Formula**: `(x - μ_B)/√(σ_B² + ε)`, where `μ_B`, `σ_B` are batch mean/variance.
  - **When**: Used in CNNs, feedforward networks.
  - **Pros**: Reduces internal covariate shift, stabilizes training.
  - **Cons**: Depends on batch size, issues at inference (uses running averages).

- **Layer Normalization (LN)**:
  - **How**: Normalizes activations across all features for each sample.
  - **Formula**: `(x - μ_L)/√(σ_L² + ε)`, where `μ_L`, `σ_L` are per-sample mean/variance.
  - **When**: Used in RNNs, transformers (e.g., BERT).
  - **Pros**: Batch-independent, consistent at training/inference.
  - **Cons**: Less effective for CNNs with spatial structure.

- **Key Differences**:
  - **Scope**: BN normalizes over batch; LN over features.
  - **Use Case**: BN for CNNs; LN for sequential models.
  - **Batch Size**: BN struggles with small batches; LN doesn’t.

**Example**:
- CNN: Use BN to normalize image channels.
- Transformer: Use LN for word embeddings.

**Interview Tips**:
- Clarify scope: “BN uses batch stats, LN uses sample stats.”
- Mention trade-offs: “BN needs large batches; LN is more flexible.”
- Relate to models: “Transformers favor LN for stability.”

---

## 6. What are convolutional neural networks (CNNs) and when are they used?

**Answer**:

**Convolutional Neural Networks (CNNs)** are specialized neural networks for grid-like data (e.g., images, time series), using convolutional layers to extract spatial or temporal features.

- **How They Work**:
  - **Convolutional Layers**: Apply filters to input (e.g., image) to detect features (edges, textures).
  - **Pooling Layers**: Downsample feature maps (e.g., max pooling) to reduce size, preserve key information.
  - **Fully Connected Layers**: Combine features for final prediction (e.g., class).
  - **Non-Linearities**: ReLU after convolutions for flexibility.

- **When Used**:
  - **Computer Vision**: Image classification, object detection, segmentation.
  - **Time Series**: 1D convolutions for signal processing.
  - **Structured Data**: When spatial relationships matter (e.g., grid layouts).

- **Advantages**:
  - Parameter sharing (filters) reduces computation.
  - Captures local patterns (e.g., edges → objects).
  - Translation-invariant (same filter everywhere).

**Example**:
- Task: Classify cats vs. dogs.
- CNN: Convolutions detect fur, eyes; pooling reduces size; output predicts class.

**Interview Tips**:
- Highlight filters: “Convolutions learn patterns like edges.”
- Mention pooling: “Reduces size, prevents overfitting.”
- Be ready to sketch: “Input → Conv → Pool → FC.”

---

## 7. Explain the concept of recurrent neural networks (RNNs)

**Answer**:

**Recurrent Neural Networks (RNNs)** are designed for sequential data, maintaining a hidden state to capture temporal dependencies.

- **How They Work**:
  - **Structure**: Each time step processes an input and updates a hidden state.
  - **Math**: `h_t = f(W_h * h_{t-1} + W_x * x_t + b)`, where `f` is activation (e.g., tanh).
  - **Output**: Predict at each step or final step (e.g., sequence label).
  - **Backpropagation Through Time (BPTT)**: Unroll network, compute gradients across steps.

- **Use Cases**:
  - NLP: Sentiment analysis, language modeling.
  - Time Series: Stock price prediction.
  - Speech: Transcription.

- **Limitations**:
  - Vanishing gradients: Hard to learn long-term dependencies.
  - Slow training: Sequential computation.

**Example**:
- Task: Predict next word in “I am a…”.
- RNN: Processes “I”, “am”, “a”, predicts “cat” based on hidden state.

**Interview Tips**:
- Emphasize sequence: “RNNs remember past inputs via hidden state.”
- Mention limits: “Struggles with long sequences.”
- Be ready to compare: “LSTM fixes RNN’s gradient issues.”

---

## 8. What are LSTMs and how do they address the limitations of RNNs?

**Answer**:

**Long Short-Term Memory (LSTM)** networks are a type of RNN designed to model long-term dependencies by addressing vanishing gradients.

- **How They Work**:
  - **Gates**:
    - **Forget Gate**: Decides what to discard from memory (`f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)`).
    - **Input Gate**: Decides what to add (`i_t = sigmoid`, `C_t’ = tanh`).
    - **Output Gate**: Decides output (`o_t = sigmoid`, `h_t = o_t * tanh(C_t)`).
  - **Cell State**: Maintains long-term memory, updated via gates.
  - **Result**: Selectively remembers or forgets information.

- **Addressing RNN Limitations**:
  - **Vanishing Gradients**: Gates allow gradients to flow through cell state, learning long sequences.
  - **Short-Term Bias**: Cell state retains information over many steps (e.g., 100+ tokens).
  - **Stability**: Controlled updates prevent exploding gradients.

**Example**:
- RNN: Fails to predict “is” in “The sky … blue, it is” (long gap).
- LSTM: Remembers “sky” via cell state, predicts “is”.

**Interview Tips**:
- Explain gates: “Forget gate decides what to keep or drop.”
- Compare: “Unlike RNNs, LSTMs handle long-term memory.”
- Be ready to sketch: “Show cell state and gates.”

---

## 9. What is the difference between a generative and discriminative model?

**Answer**:

- **Discriminative Model**:
  - **Goal**: Model conditional probability `P(y|x)` (predict `y` given `x`).
  - **How**: Learns decision boundary between classes.
  - **Examples**:
    - Logistic regression, SVM, neural networks for classification.
    - CNN for image classification.
  - **Use Case**: Classification, regression.

- **Generative Model**:
  - **Goal**: Model joint probability `P(x,y)` or `P(x)` (generate `x` and `y`).
  - **How**: Learns data distribution, can sample new data.
  - **Examples**:
    - Naive Bayes, GANs, VAEs.
    - GPT for text generation.
  - **Use Case**: Data generation, density estimation.

- **Key Differences**:
  - **Task**: Discriminative predicts labels; generative models data.
  - **Output**: Discriminative gives probabilities; generative samples data.
  - **Data Needs**: Generative often needs more data to model distribution.

**Example**:
- Discriminative: CNN classifies cat vs. dog.
- Generative: GAN generates cat images.

**Interview Tips**:
- Clarify math: “Discriminative models P(y|x), generative P(x,y).”
- Mention hybrids: “Some models combine both (e.g., semi-supervised).”
- Be ready to compare: “Discriminative is simpler for classification.”

---

## 10. Explain the concept of transfer learning in deep learning

**Answer**:

**Transfer learning** uses a pretrained model (trained on a large dataset) for a new task, leveraging learned features to save time and data.

- **How It Works**:
  - **Pretrained Model**: Train on large dataset (e.g., ImageNet for CNNs, BERT for NLP).
  - **Fine-Tuning**:
    - Freeze early layers (e.g., feature extractors).
    - Train later layers or new head on task-specific data.
  - **Full Fine-Tuning** (optional): Unfreeze all layers, train with small learning rate.

- **Benefits**:
  - **Less Data**: Works with small datasets.
  - **Faster Training**: Avoids training from scratch.
  - **Better Performance**: Leverages general features (e.g., edges, word semantics).

- **Use Cases**:
  - **Vision**: Use ResNet for medical imaging with few samples.
  - **NLP**: Fine-tune BERT for sentiment analysis.
  - **Audio**: Pretrained models for speech recognition.

**Example**:
- Task: Classify flowers.
- Action: Use pretrained ResNet, fine-tune last layer on 100 flower images.
- Result: 90% accuracy vs. 60% from scratch.

**Interview Tips**:
- Emphasize efficiency: “Saves time and data.”
- Discuss layers: “Early layers learn generic features, later ones task-specific.”
- Be ready to suggest models: “ResNet for vision, BERT for text.”

---

## 11. What are attention mechanisms and how are they used in transformers?

**Answer**:

**Attention mechanisms** allow models to focus on relevant parts of the input when making predictions, improving performance on sequential tasks.

- **How They Work**:
  - **Scaled Dot-Product Attention**:
    - Compute similarity between query (`Q`) and key (`K`) vectors: `Attention = softmax(QK^T/√d_k)V`.
    - Weight value (`V`) vectors by similarity, producing context-aware output.
  - **Multi-Head Attention**: Run multiple attention layers in parallel, concatenate outputs.
  - **Self-Attention**: Input attends to itself (e.g., words in a sentence).

- **In Transformers**:
  - **Encoder**: Self-attention learns relationships between input tokens (e.g., words).
  - **Decoder**: Attends to encoder output and previous tokens for generation.
  - **Benefits**:
    - Captures long-range dependencies (unlike RNNs).
    - Parallelizable (unlike sequential RNNs).

**Example**:
- Task: Translate “I love cats” to French.
- Attention: Focuses on “cats” when predicting “chats,” ignoring irrelevant tokens.

**Interview Tips**:
- Explain intuition: “Attention weighs important inputs dynamically.”
- Mention transformers: “Core to BERT, GPT.”
- Be ready to sketch: “Show Q, K, V computation.”

---

## 12. What is the difference between SGD, Adam, and RMSprop optimizers?

**Answer**:

- **SGD (Stochastic Gradient Descent)**:
  - **How**: Updates weights using gradient of loss: `w = w - lr * gradient`.
  - **Pros**: Simple, stable with proper tuning.
  - **Cons**: Slow convergence, sensitive to learning rate.
  - **Variants**: Momentum (accelerates gradients).

- **RMSprop**:
  - **How**: Adapts learning rate per parameter using moving average of squared gradients.
  - **Formula**: `E[g²]_t = β * E[g²]_{t-1} + (1-β) * g²`, `w = w - lr * g/√(E[g²] + ε)`.
  - **Pros**: Handles non-stationary objectives, faster than SGD.
  - **Cons**: Still requires tuning, less adaptive than Adam.

- **Adam**:
  - **How**: Combines momentum and RMSprop, using moving averages of gradients and squared gradients.
  - **Formula**: `m_t = β1 * m_{t-1} + (1-β1) * g`, `v_t = β2 * v_{t-1} + (1-β2) * g²`, `w = w - lr * m_t/√(v_t + ε)`.
  - **Pros**: Adaptive, fast convergence, robust defaults.
  - **Cons**: Can overfit, complex to tune.

**Example**:
- SGD: Slow for deep CNN, needs tuning.
- Adam: Fast for transformer, good out-of-box.

**Interview Tips**:
- Highlight adaptivity: “Adam adjusts learning rates automatically.”
- Discuss trade-offs: “SGD is simple but slow; Adam is fast but tricky.”
- Mention use case: “Adam for most DL tasks.”

---

## 13. What are GANs and how do they work?

**Answer**:

**Generative Adversarial Networks (GANs)** are models that generate realistic data by pitting two networks against each other.

- **Components**:
  - **Generator**: Takes random noise, produces fake data (e.g., images).
  - **Discriminator**: Classifies data as real (from dataset) or fake (from generator).
- **Training**:
  - **Adversarial Game**:
    - Generator tries to fool discriminator.
    - Discriminator tries to distinguish real vs. fake.
  - **Loss**:
    - Discriminator: Binary cross-entropy (real vs. fake).
    - Generator: Minimize discriminator’s confidence in fake data.
  - **Optimization**: Alternate training, balance convergence.
- **Use Cases**: Image generation, style transfer, data augmentation.

**Example**:
- Task: Generate faces.
- Generator: Creates face-like images from noise.
- Discriminator: Improves by rejecting fakes, forcing generator to refine.

**Interview Tips**:
- Explain game: “Generator and discriminator compete.”
- Mention challenges: “Training is unstable, needs careful tuning.”
- Be ready to sketch: “Noise → Generator → Discriminator.”

---

## 14. What are autoencoders and what are they used for?

**Answer**:

**Autoencoders** are neural networks that learn to compress and reconstruct data, useful for unsupervised tasks.

- **Structure**:
  - **Encoder**: Maps input to lower-dimensional latent space (`z = f(x)`).
  - **Decoder**: Reconstructs input from latent space (`x’ = g(z)`).
  - **Loss**: Reconstruction error (e.g., MSE: `||x - x’||²`).
- **Training**: Minimize difference between input and output.

- **Use Cases**:
  - **Dimensionality Reduction**: Like PCA but non-linear.
  - **Denoising**: Reconstruct clean data from noisy input.
  - **Anomaly Detection**: High reconstruction error flags outliers.
  - **Feature Learning**: Extract features for downstream tasks.

**Example**:
- Task: Compress images.
- Autoencoder: Encodes 784 pixels to 32-dim, reconstructs with low error.

**Interview Tips**:
- Compare to PCA: “Autoencoders are non-linear, more flexible.”
- Mention variants: “VAEs add probabilistic latent space.”
- Be ready to sketch: “Input → Encoder → Latent → Decoder.”

---

## 15. How do you prevent overfitting in deep neural networks?

**Answer**:

Overfitting in deep networks occurs when the model memorizes training data. Techniques to prevent it:

- **Regularization**:
  - **L1/L2**: Penalize large weights.
  - **Dropout**: Randomly drop neurons during training (e.g., 20% rate).
- **Data Augmentation**:
  - Add variations (e.g., rotate images, synonym replacement in text).
- **Simplify Model**:
  - Reduce layers or neurons.
- **Batch Normalization**:
  - Stabilizes training, acts as regularizer.
- **Early Stopping**:
  - Stop training when validation loss stops improving.
- **More Data**:
  - Collect or generate more training samples.
- **Cross-Validation**:
  - Use k-fold CV to tune hyperparameters.

**Example**:
- CNN overfits (train acc 95%, test 70%).
- Action: Add dropout (0.5), augment images, early stop.
- Result: Test acc 85%.

**Interview Tips**:
- Prioritize: “Dropout and augmentation are go-to methods.”
- Relate to data: “More data is best but not always feasible.”
- Suggest tuning: “I’d monitor validation loss closely.”

---

## 16. What is the role of pooling layers in CNNs?

**Answer**:

**Pooling layers** in CNNs downsample feature maps, reducing spatial dimensions while preserving key information.

- **Role**:
  - **Reduce Computation**: Smaller maps speed up training/inference.
  - **Prevent Overfitting**: Fewer parameters reduce complexity.
  - **Translation Invariance**: Pooling (e.g., max) focuses on dominant features, ignoring small shifts.

- **Types**:
  - **Max Pooling**: Take maximum value in a region (e.g., 2x2 window).
  - **Average Pooling**: Take average value.
  - **Global Pooling**: Reduce entire map to single value (e.g., for classification).

**Example**:
- Input: 32x32 feature map.
- Max Pool (2x2, stride 2): Output 16x16, keeps strongest features.

**Interview Tips**:
- Explain intuition: “Pooling summarizes regions, like zooming out.”
- Discuss trade-offs: “Max pooling is aggressive, average is smoother.”
- Be ready to sketch: “Show 2x2 max pool.”

---

## 17. What is the difference between 1x1 convolution and regular convolution?

**Answer**:

- **Regular Convolution**:
  - **How**: Applies a filter (e.g., 3x3) to input, computing weighted sums over a spatial region.
  - **Purpose**: Detects spatial patterns (e.g., edges, textures).
  - **Output**: Preserves spatial dimensions (with padding) or reduces them.

- **1x1 Convolution**:
  - **How**: Applies a 1x1 filter, acting like a dense layer across channels.
  - **Purpose**:
    - Dimensionality reduction (e.g., reduce 256 channels to 64).
    - Learn cross-channel interactions.
    - Add non-linearity (with activation).
  - **Output**: Same spatial size, different channel depth.

- **Key Differences**:
  - **Scope**: Regular conv captures spatial patterns; 1x1 focuses on channels.
  - **Parameters**: 1x1 is lightweight (e.g., fewer weights).
  - **Use Case**: 1x1 in bottlenecks (e.g., Inception, ResNet).

**Example**:
- Regular: 3x3 conv detects edges in image.
- 1x1: Reduces 512 channels to 128, saves computation.

**Interview Tips**:
- Clarify role: “1x1 is like a per-pixel dense layer.”
- Mention efficiency: “Used in ResNet for bottlenecks.”
- Be ready to compute: “Show parameter savings.”

---

## 18. What are residual connections and why are they important in deep networks?

**Answer**:

**Residual connections** (or skip connections) allow a layer to learn the residual (difference) between input and output, easing training in deep networks.

- **How They Work**:
  - Instead of `y = F(x)` (direct mapping), layer learns `y = F(x) + x`.
  - **Math**: Output = `F(x, W) + x`, where `F` is the layer’s function.
  - Implemented in ResNet: Add input to output before activation.

- **Why Important**:
  - **Mitigate Vanishing Gradients**: Direct path ensures gradients flow to early layers.
  - **Enable Deeper Networks**: ResNet-50, ResNet-100 possible without degradation.
  - **Identity Learning**: If `F(x) = 0`, layer learns identity, avoiding harm.

**Example**:
- Deep CNN without residuals: Accuracy drops after 20 layers.
- With residuals: ResNet-152 maintains high accuracy.

**Interview Tips**:
- Explain intuition: “Layers learn what to adjust, not everything.”
- Mention ResNet: “Key to scaling deep networks.”
- Be ready to sketch: “Show x → F(x) → +x.”

---

## 19. What is the difference between word2vec and GloVe embeddings?

**Answer**:

- **Word2Vec**:
  - **How**: Predicts words given context (CBOW) or context given word (Skip-gram) using shallow neural network.
  - **Training**: Local context (e.g., 5-word window), optimizes likelihood.
  - **Pros**: Captures local semantics, fast training.
  - **Cons**: Ignores global statistics, less effective for rare words.

- **GloVe**:
  - **How**: Factorizes word co-occurrence matrix to learn embeddings, weighted by frequency.
  - **Training**: Global statistics (e.g., how often words co-occur across corpus).
  - **Pros**: Captures global relationships, better for rare words.
  - **Cons**: Requires precomputed matrix, less flexible.

- **Key Differences**:
  - **Scope**: Word2Vec uses local context; GloVe uses global co-occurrences.
  - **Training**: Word2Vec is predictive; GloVe is count-based.
  - **Performance**: GloVe often better for semantic tasks; Word2Vec for syntactic.

**Example**:
- Word2Vec: “king - man + woman ≈ queen” (local analogy).
- GloVe: Captures “king” and “queen” similarity globally.

**Interview Tips**:
- Clarify method: “Word2Vec predicts, GloVe factorizes.”
- Mention use case: “GloVe for downstream NLP tasks.”
- Be ready to compare: “Both learn dense vectors but differ in data.”

---

## 20. What is the role of positional encoding in transformers?

**Answer**:

**Positional encoding** in transformers adds information about token positions in a sequence, since self-attention is permutation-invariant.

- **Why Needed**:
  - Transformers process tokens simultaneously (unlike RNNs).
  - Without encoding, “I love cats” and “Cats love I” are identical.
- **How It Works**:
  - Add fixed or learned vectors to token embeddings.
  - **Fixed (BERT)**: Use sine/cosine functions: `PE(pos, 2i) = sin(pos/10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)`.
  - **Learned**: Train position embeddings as parameters.
- **Role**:
  - Encodes order, enabling transformers to understand sequence structure.
  - Allows generalization to longer sequences (fixed encodings).

**Example**:
- Input: “I love cats”.
- Encoding: Each word’s embedding + positional vector (e.g., pos=1, 2, 3).
- Result: Transformer captures word order for translation.

**Interview Tips**:
- Explain invariance: “Attention doesn’t know order without encoding.”
- Mention sine/cosine: “Fixed patterns generalize well.”
- Be ready to sketch: “Show embedding + positional vector.”

---

## Notes

- **Depth**: Answers dive into neural network specifics, ideal for deep learning interviews.
- **Clarity**: Explanations are structured for verbal delivery, with examples and trade-offs.
- **Practicality**: Includes real-world applications (e.g., vision, NLP) and implementation tips (e.g., fine-tuning).
- **Consistency**: Matches the style of previous files for a cohesive repository.

For deeper practice, implement a CNN or transformer (see [ML Coding](ml-coding.md)) or explore [ML System Design](ml-system-design.md) for scaling deep learning solutions. 🚀

---

**Next Steps**: Build on these skills with [Statistics & Probability](statistics-probability.md) for foundational math or revisit [Tree-Based Models](tree-based-models.md) for ensemble methods! 🌟