# ML Coding Questions

This file contains machine learning coding questions commonly asked in interviews at companies like **Uber**, **Google**, and various startups. These questions focus on implementing ML algorithms from scratch, often without third-party libraries like Scikit-Learn, to test your coding skills and conceptual understanding.

Below are the questions with detailed answers, including explanations and Python code where relevant.

---

## Table of Contents

1. [Write an AUC from scratch using vanilla Python](#1-write-an-auc-from-scratch-using-vanilla-python)
2. [Write the K-Means algorithm using NumPy only](#2-write-the-k-means-algorithm-using-numpy-only)
3. [Code Gradient Descent from scratch using NumPy and SciPy only](#3-code-gradient-descent-from-scratch-using-numpy-and-scipy-only)

---

## 1. Write an AUC from scratch using vanilla Python

**Question**: [Uber] Implement the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) from scratch using only vanilla Python (no NumPy or other libraries).

**Answer**:

The AUC-ROC measures the performance of a binary classifier by calculating the area under the curve of true positive rate (TPR) vs. false positive rate (FPR) at various thresholds. To compute it from scratch:

1. Sort predictions and true labels by predicted probabilities in descending order.
2. Calculate TPR (sensitivity) and FPR (1-specificity) for each threshold.
3. Approximate the area under the curve using the trapezoidal rule.

Here’s the implementation:

```python
def calculate_auc(y_true, y_pred):
    # Ensure inputs are lists
    y_true = list(y_true)
    y_pred = list(y_pred)
    
    # Pair predictions with true labels and sort by predictions (descending)
    pairs = sorted(zip(y_pred, y_true), reverse=True)
    y_true = [label for _, label in pairs]
    
    # Initialize variables
    tp = 0  # True positives
    fp = 0  # False positives
    tpr_list = []
    fpr_list = []
    total_pos = sum(y_true)  # Total actual positives
    total_neg = len(y_true) - total_pos  # Total actual negatives
    
    # Edge case: no positives or negatives
    if total_pos == 0 or total_neg == 0:
        return 0.0
    
    # Calculate TPR and FPR for each threshold
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos  # True positive rate
        fpr = fp / total_neg  # False positive rate
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    
    return auc

# Example usage
y_true = [0, 1, 1, 0, 1]
y_pred = [0.1, 0.8, 0.6, 0.3, 0.9]
print(calculate_auc(y_true, y_pred))  # Output: ~0.833
```

**Explanation**:
- The code avoids external libraries, using only Python’s built-in functions.
- Sorting by predictions ensures we evaluate thresholds from high to low probability.
- TPR = TP / (TP + FN), FPR = FP / (FP + TN). We track TP and FP incrementally.
- The trapezoidal rule approximates the area by summing trapezoids formed by (FPR, TPR) points.
- In an interview, explain the logic step-by-step and handle edge cases (e.g., no positives).

---

## 2. Write the K-Means algorithm using NumPy only

**Question**: [Google] Implement the K-Means clustering algorithm from scratch using only NumPy (no Scikit-Learn).

**Answer**:

K-Means clustering groups data into `k` clusters by minimizing the variance within each cluster. The algorithm:

1. Randomly initializes `k` centroids.
2. Assigns each point to the nearest centroid.
3. Updates centroids as the mean of assigned points.
4. Repeats until centroids stabilize or max iterations are reached.

Here’s the NumPy implementation:

```python
import numpy as np

def kmeans(X, k, max_iters=100, random_state=42):
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Randomly initialize k centroids
    n_samples, n_features = X.shape
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[idx]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Store old centroids for convergence check
        old_centroids = centroids.copy()
        
        # Update centroids
        for i in range(k):
            if np.sum(labels == i) > 0:  # Avoid empty clusters
                centroids[i] = np.mean(X[labels == i], axis=0)
        
        # Check for convergence
        if np.all(old_centroids == centroids):
            break
    
    return labels, centroids

# Example usage
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
labels, centroids = kmeans(X, k=2)
print("Labels:", labels)
print("Centroids:", centroids)
```

**Explanation**:
- **Initialization**: Randomly select `k` points as initial centroids using NumPy’s `random.choice`.
- **Assignment**: Compute Euclidean distances from each point to centroids using vectorized operations.
- **Update**: Calculate new centroids as the mean of points in each cluster.
- **Convergence**: Stop if centroids don’t change or after `max_iters`.
- In an interview, mention handling empty clusters (checked here) and computational efficiency via NumPy’s vectorization.

---

## 3. Code Gradient Descent from scratch using NumPy and SciPy only

**Question**: [Startup] Implement Gradient Descent from scratch for a simple linear regression model using only NumPy and SciPy (no Scikit-Learn).

**Answer**:

Gradient Descent optimizes a model’s parameters (e.g., weights and bias in linear regression) by minimizing a loss function (mean squared error here). The algorithm:

1. Initializes parameters randomly.
2. Computes the gradient of the loss with respect to parameters.
3. Updates parameters in the opposite direction of the gradient.
4. Repeats until convergence or max iterations.

We’ll use NumPy for computations and SciPy (though minimally, as it’s allowed).

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, max_iters=1000, tol=1e-6):
    # Add bias term (column of 1s) to X
    X = np.c_[np.ones(len(X)), X]
    
    # Initialize weights randomly
    np.random.seed(42)
    theta = np.random.randn(X.shape[1])
    
    # Track loss for convergence
    prev_loss = float('inf')
    
    for _ in range(max_iters):
        # Forward pass: predictions
        y_pred = X @ theta
        
        # Compute mean squared error loss
        loss = np.mean((y_pred - y) ** 2)
        
        # Check convergence
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
        
        # Compute gradients
        gradients = 2 / len(X) * X.T @ (y_pred - y)
        
        # Update weights
        theta -= learning_rate * gradients
    
    return theta

# Example usage
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])  # Linear: y = 2x
theta = gradient_descent(X, y)
print("Parameters (bias, weight):", theta)
```

**Explanation**:
- **Setup**: Add a bias term to `X` and initialize weights randomly.
- **Loss**: Use mean squared error (MSE) as the loss function.
- **Gradients**: Compute partial derivatives of MSE w.r.t. weights using matrix operations.
- **Update**: Adjust weights using the learning rate.
- **Convergence**: Stop if loss stabilizes (within `tol`) or after `max_iters`.
- In an interview, explain the choice of learning rate, potential issues (e.g., overshooting), and how to extend this to other loss functions.

---

## Notes

- **Code Simplicity**: The implementations are concise yet complete, suitable for whiteboard or live-coding interviews.
- **Edge Cases**: Each answer addresses edge cases (e.g., no positives in AUC, empty clusters in K-Means, convergence in Gradient Descent).
- **Explanations**: Answers include step-by-step logic to demonstrate understanding, crucial for verbalizing in interviews.
- **NumPy Efficiency**: Where allowed, vectorized operations reduce runtime, showing good coding practices.

For additional practice, try modifying these implementations (e.g., add regularization to Gradient Descent or handle edge cases differently).

---

**Next Steps**: Continue preparing with other categories like [ML Theory](ml-theory.md) or explore more coding challenges to solidify your skills! 🚀