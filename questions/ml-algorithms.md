# ML Algorithms (Depth) Questions

This file contains machine learning algorithm questions commonly asked in interviews at companies like **Amazon**. These questions assess your **in-depth understanding** of specific algorithms, focusing on their mechanics, assumptions, trade-offs, and differences. They test your ability to articulate detailed knowledge beyond general ML concepts.

Below are the questions with comprehensive answers, including explanations, mathematical intuition where relevant, and practical insights for interviews.

---

## Table of Contents

1. [What is the pseudocode of the Random Forest model?](#1-what-is-the-pseudocode-of-the-random-forest-model)
2. [What is the variance and bias of the Random Forest model?](#2-what-is-the-variance-and-bias-of-the-random-forest-model)
3. [How is the Random Forest different from Gradient Boosted Trees?](#3-how-is-the-random-forest-different-from-gradient-boosted-trees)

---

## 1. What is the pseudocode of the Random Forest model?

**Question**: [Amazon] Provide the pseudocode for the Random Forest algorithm.

**Answer**:

**Random Forest** is an ensemble learning method that combines multiple decision trees to improve predictive accuracy and reduce overfitting. It uses **bagging** (Bootstrap Aggregating) and **feature randomness** to create diverse trees, then aggregates their predictions (majority vote for classification, average for regression).

**Pseudocode**:

Algorithm: RandomForest(X, y, n_trees, max_depth, min_samples_split, max_features)
    Input:
        X: feature matrix (n_samples, n_features)
        y: target vector (n_samples)
        n_trees: number of trees
        max_depth: maximum depth of each tree (optional)
        min_samples_split: minimum samples to split a node
        max_features: number of features to consider at each split
    Output:
        forest: list of trained decision trees

    Initialize:
        forest = []

    For i = 1 to n_trees:
        # Bootstrap sampling: sample n_samples with replacement
        indices = RandomSampleWithReplacement(n_samples)
        X_boot = X[indices]
        y_boot = y[indices]

        # Build a decision tree with feature randomness
        tree = DecisionTree(max_depth, min_samples_split)
        For each node in tree:
            If node satisfies stopping criteria (e.g., max_depth, min_samples_split):
                Make node a leaf, assign majority class (classification) or mean (regression)
            Else:
                # Randomly select max_features features
                candidate_features = RandomSubset(features, max_features)
                # Find best split based on criterion (e.g., Gini, entropy, MSE)
                best_feature, best_threshold = FindBestSplit(X_boot, y_boot, candidate_features)
                Split node into left and right children
                Recurse on children
        Add tree to forest

    Return forest

Prediction:
    For a new sample x:
        predictions = []
        For each tree in forest:
            prediction = Predict(tree, x)
            Append prediction to predictions
        If classification:
            Return majority vote of predictions
        If regression:
            Return mean of predictions

**Explanation**:
- **Bootstrap Sampling**: Each tree is trained on a random subset of the data (with replacement), ensuring diversity.
- **Feature Randomness**: At each split, only a subset of features (`max_features`) is considered, reducing correlation between trees.
- **Decision Tree**: Uses standard splitting criteria (e.g., Gini impurity for classification, MSE for regression).
- **Aggregation**: Combines predictions to stabilize results and improve accuracy.

**Interview Tips**:
- Clarify parameters like `max_features` (often `sqrt(n_features)` for classification, `n_features/3` for regression).
- Mention stopping criteria (e.g., `max_depth`, `min_samples_split`) to prevent overfitting.
- Explain how bagging reduces variance compared to a single tree.
- Be ready to sketch a tree split or discuss Gini/entropy if asked for details.

---

## 2. What is the variance and bias of the Random Forest model?

**Question**: [Amazon] Discuss the bias and variance characteristics of the Random Forest model.

**Answer**:

**Bias** and **variance** describe a model’s error components, and Random Forest’s ensemble nature gives it distinct characteristics:

- **Bias**:
  - **Definition**: Bias measures how much a model’s predictions deviate from true values due to oversimplification.
  - **Random Forest Bias**: Random Forest has **similar bias** to a single decision tree because each tree is grown deep (low bias) to capture complex patterns. However, since trees are unpruned or lightly constrained (e.g., limited `max_depth`), individual trees have low bias, and averaging doesn’t significantly increase it.
  - **Example**: For a non-linear dataset, Random Forest fits complex boundaries (low bias) unless restricted by parameters like shallow depth.

- **Variance**:
  - **Definition**: Variance measures how sensitive a model’s predictions are to changes in the training data.
  - **Random Forest Variance**: Random Forest **reduces variance** compared to a single decision tree. A single tree overfits to training data noise (high variance), but Random Forest’s bagging and feature randomness create diverse trees. Averaging (regression) or voting (classification) smooths out individual tree errors, lowering overall variance.
  - **Example**: If one tree overpredicts due to noise, others balance it out, stabilizing predictions.

- **Bias-Variance Trade-off**:
  - Random Forest achieves **low bias** (like deep trees) and **low variance** (via ensemble averaging), making it robust for many tasks.
  - Total error = (Bias)² + Variance + Irreducible Error. Random Forest minimizes variance while maintaining low bias, reducing total error compared to a single tree.

- **Parameter Impact**:
  - **More trees (`n_estimators`)**: Further reduces variance without affecting bias much.
  - **Smaller `max_features`**: Increases diversity, reducing variance but potentially increasing bias if too small.
  - **Shallow trees (`max_depth`)**: Increases bias, reduces variance slightly.

**Practical Example**:
- On a noisy classification dataset:
  - A single decision tree might have low bias (fits training data well) but high variance (sensitive to noise).
  - A Random Forest with 100 trees has similar bias (still captures patterns) but lower variance (stable predictions across datasets).

**Interview Tips**:
- Contrast with a single tree: “A single tree has low bias but high variance; Random Forest keeps low bias and reduces variance through bagging.”
- Mention that bias depends on tree depth and data complexity.
- Be ready to discuss how hyperparameters tune the trade-off (e.g., `n_estimators`, `max_features`).
- Sketch the bias-variance curve if asked, showing Random Forest’s advantage.

---

## 3. How is the Random Forest different from Gradient Boosted Trees?

**Question**: [Amazon] Explain the key differences between Random Forest and Gradient Boosted Trees.

**Answer**:

**Random Forest** and **Gradient Boosted Trees** are both ensemble methods that combine decision trees, but they differ in how trees are built and combined, leading to distinct strengths and weaknesses.

1. **Methodology**:
   - **Random Forest**:
     - Uses **bagging** (Bootstrap Aggregating).
     - Trains multiple trees **independently** on random subsets of data (with replacement) and features.
     - Combines predictions via **majority vote** (classification) or **averaging** (regression).
     - Goal: Reduce variance by averaging uncorrelated trees.
   - **Gradient Boosted Trees**:
     - Uses **boosting**.
     - Trains trees **sequentially**, where each tree corrects errors of previous ones by fitting to the residual errors (gradients of the loss function).
     - Combines predictions via **weighted sum** of trees.
     - Goal: Reduce bias and variance by iteratively improving the model.

2. **Tree Construction**:
   - **Random Forest**:
     - Trees are typically **deep** (low bias) to capture complex patterns.
     - Uses **feature randomness** (`max_features`) to ensure diversity.
     - Each tree is trained on a bootstrap sample, reducing correlation.
   - **Gradient Boosted Trees**:
     - Trees are usually **shallow** (weak learners) to avoid overfitting.
     - All features are considered at each split (unless specified), focusing on error correction.
     - Uses the full dataset (or subsamples) with weights adjusted per iteration.

3. **Bias and Variance**:
   - **Random Forest**:
     - **Low bias**: Deep trees fit data well.
     - **Low variance**: Bagging averages out errors from uncorrelated trees.
     - Better at handling noisy data due to independence.
   - **Gradient Boosted Trees**:
     - **Lower bias**: Sequential correction refines predictions, capturing complex patterns.
     - **Higher variance**: Sequential dependence makes it sensitive to noise unless regularized.
     - More prone to overfitting without tuning.

4. **Training Speed**:
   - **Random Forest**:
     - **Faster**: Trees are trained in parallel, and computation scales well.
     - Less sensitive to hyperparameters, easier to tune.
   - **Gradient Boosted Trees**:
     - **Slower**: Sequential training means each tree waits for the previous one.
     - Requires careful tuning (e.g., learning rate, tree depth).

5. **Performance**:
   - **Random Forest**:
     - Excels in **general-purpose tasks** with noisy or high-dimensional data.
     - Robust out-of-the-box, less tuning needed.
   - **Gradient Boosted Trees**:
     - Often **outperforms** Random Forest on structured/tabular data with careful tuning.
     - Preferred in competitions (e.g., XGBoost, LightGBM) for maximizing predictive accuracy.

6. **Hyperparameters**:
   - **Random Forest**:
     - Key parameters: `n_estimators`, `max_features`, `max_depth`.
     - Less sensitive to overfitting.
   - **Gradient Boosted Trees**:
     - Key parameters: `n_estimators`, `learning_rate`, `max_depth`, `subsample`.
     - Requires balancing `learning_rate` and `n_estimators` to avoid overfitting.

**Practical Example**:
- **Fraud Detection**:
  - **Random Forest**: Good for noisy, high-dimensional data; quick to train and robust.
  - **Gradient Boosted Trees**: Better if tuned to focus on rare fraud cases, but needs regularization to avoid overfitting.
- **Choice**: Use Random Forest for quick prototyping; use Gradient Boosting (e.g., XGBoost) for maximizing accuracy with tuning.

**Interview Tips**:
- Emphasize **bagging vs. boosting** as the core difference.
- Compare strengths: “Random Forest is robust and fast; Gradient Boosting is powerful but needs tuning.”
- Mention popular implementations (e.g., Random Forest in Scikit-Learn, XGBoost/LightGBM for boosting).
- Be ready to discuss when to use each (e.g., Random Forest for noisy data, Gradient Boosting for structured data).

---

## Notes

- **Depth**: Answers dive into algorithmic details, suitable for “depth” rounds where interviewers expect thorough knowledge.
- **Clarity**: Explanations are structured to articulate complex ideas simply, ideal for verbalizing in interviews.
- **Practicality**: Includes examples and trade-offs to show real-world application (e.g., Random Forest vs. Gradient Boosting use cases).
- **Consistency**: Matches the style of `ml-coding.md` and `ml-theory.md` for a cohesive repository.

For deeper practice, implement Random Forest or Gradient Boosting manually (see [ML Coding](ml-coding.md)) or explore related topics in [Applied ML Cases](applied-ml-cases.md). 🚀

---

**Next Steps**: Strengthen your prep with [ML Theory](ml-theory.md) for broader concepts or dive into [Applied ML Cases](applied-ml-cases.md) for business-focused problems! 🌟