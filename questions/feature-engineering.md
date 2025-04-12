# Feature Engineering & Data Preprocessing Questions

This file contains feature engineering and data preprocessing questions commonly asked in interviews at companies like **Google**, **Amazon**, and others. These questions assess your ability to **transform raw data** into meaningful features and prepare it for machine learning models. They test your understanding of handling missing data, categorical variables, feature selection, and more.

Below are the questions with detailed answers, including explanations, practical techniques, and insights for interviews.

---

## Table of Contents

1. [How do you handle missing or corrupted data in a dataset?](#1-how-do-you-handle-missing-or-corrupted-data-in-a-dataset)
2. [How would you handle an imbalanced dataset?](#2-how-would-you-handle-an-imbalanced-dataset)
3. [Can you explain the concept of "feature selection" in machine learning?](#3-can-you-explain-the-concept-of-feature-selection-in-machine-learning)
4. [How do you handle categorical variables in your dataset?](#4-how-do-you-handle-categorical-variables-in-your-dataset)
5. [How do filtering and wrapper methods work in feature selection?](#5-how-do-filtering-and-wrapper-methods-work-in-feature-selection)
6. [Describe a situation where you had to handle missing data. What techniques did you use?](#6-describe-a-situation-where-you-had-to-handle-missing-data-what-techniques-did-you-use)
7. [What is principal component analysis (PCA) and when is it used?](#7-what-is-principal-component-analysis-pca-and-when-is-it-used)
8. [What's the difference between PCA vs ICA?](#8-whats-the-difference-between-pca-vs-ica)
9. [How do you handle time-based features in a machine learning model?](#9-how-do-you-handle-time-based-features-in-a-machine-learning-model)
10. [What is feature hashing? When would you use it?](#10-what-is-feature-hashing-when-would-you-use-it)
11. [How do you handle hierarchical categorical variables?](#11-how-do-you-handle-hierarchical-categorical-variables)
12. [What are embedding layers and when should you use them?](#12-what-are-embedding-layers-and-when-should-you-use-them)
13. [Explain different strategies for handling outliers in different ML algorithms](#13-explain-different-strategies-for-handling-outliers-in-different-ml-algorithms)

---

## 1. How do you handle missing or corrupted data in a dataset?

**Answer**:

Missing or corrupted data can bias models or cause errors. Here are common techniques:

- **Identify**:
  - Missing: Null values, NaN, or placeholders (e.g., -999).
  - Corrupted: Out-of-range values (e.g., negative age), inconsistent formats.
  - Use summary stats or visualization (e.g., heatmaps) to detect.

- **Handle Missing Data**:
  - **Deletion**:
    - Remove rows with missing values (if <5% of data and missing at random).
    - Pros: Simple. Cons: Loses data.
  - **Imputation**:
    - **Numerical**: Mean, median, or mode (e.g., impute missing `age` with median).
    - **Categorical**: Mode or a new category (e.g., “Unknown”).
    - **Advanced**: K-Nearest Neighbors (KNN) or regression to predict missing values.
    - Pros: Preserves data. Cons: May introduce bias.
  - **Flagging**: Add a binary feature (e.g., `is_age_missing`) to capture patterns.

- **Handle Corrupted Data**:
  - Replace with plausible values (e.g., cap negative `age` at 0).
  - Treat as missing and impute.
  - Remove if unrecoverable (e.g., gibberish text).

**Example**:
- Dataset: Customer records with 10% missing `income`.
- Action: Impute with median `income` per `job_type`, add `is_income_missing` flag.
- Result: Model trains without errors, captures missingness pattern.

**Interview Tips**:
- Ask: “Is missingness random or systematic?” (affects strategy).
- Emphasize trade-offs: “Deletion is simple but risks bias; imputation preserves data but may distort.”
- Mention domain knowledge: “I’d consult stakeholders to understand why data is missing.”

---

## 2. How would you handle an imbalanced dataset?

**Answer**:

An imbalanced dataset has unequal class distributions (e.g., 90% negative, 10% positive), biasing models toward the majority class. Techniques include:

- **Resampling**:
  - **Oversampling**: Duplicate or generate minority class samples (e.g., SMOTE creates synthetic samples).
    - Pros: Increases minority representation. Cons: Risk of overfitting.
  - **Undersampling**: Remove majority class samples randomly.
    - Pros: Faster training. Cons: Loses data, may miss patterns.

- **Class Weights**:
  - Assign higher weights to minority class in the loss function (e.g., `weight = 1/frequency`).
  - Pros: No data modification, model focuses on minority. Cons: May not suffice for extreme imbalance.

- **Data Augmentation**:
  - Generate new minority samples (e.g., perturb existing data).
  - Pros: Adds diversity. Cons: Domain-specific.

- **Anomaly Detection**:
  - Treat minority class as anomalies (e.g., Isolation Forest).
  - Pros: Effective for extreme imbalance. Cons: Requires reframing problem.

- **Evaluation Metrics**:
  - Use precision, recall, F1-score, or AUC-ROC instead of accuracy.
  - Pros: Focuses on minority class performance.

- **Collect More Data**:
  - If feasible, gather more minority class samples.
  - Pros: Addresses root cause. Cons: Often costly.

**Example**:
- Dataset: Fraud detection, 1% fraud.
- Action: Use SMOTE to oversample fraud cases, apply class weights in XGBoost, evaluate with F1-score.
- Result: Improved fraud detection (F1 = 0.7 vs. 0.4).

**Interview Tips**:
- Clarify imbalance severity: “Is it 80-20 or 99-1?”
- Discuss trade-offs: “SMOTE risks overfitting; undersampling loses data.”
- Emphasize metrics: “Accuracy misleads, so I’d use F1 or AUC.”

---

## 3. Can you explain the concept of "feature selection" in machine learning?

**Answer**:

**Feature selection** is the process of choosing a subset of relevant features to improve model performance, reduce overfitting, and speed up training.

- **Why Important**:
  - **Reduces Overfitting**: Fewer irrelevant features prevent fitting noise.
  - **Improves Speed**: Smaller feature set lowers computation.
  - **Enhances Interpretability**: Simplifies model explanations.

- **Methods**:
  - **Filter Methods**:
    - Evaluate features independently (e.g., correlation, chi-squared).
    - Example: Select top 10 features by mutual information.
    - Pros: Fast, scalable. Cons: Ignores feature interactions.
  - **Wrapper Methods**:
    - Evaluate subsets using model performance (e.g., recursive feature elimination).
    - Example: Train model with different feature sets, keep best.
    - Pros: Considers interactions. Cons: Computationally expensive.
  - **Embedded Methods**:
    - Perform selection during training (e.g., L1 regularization in Lasso).
    - Example: Lasso sets irrelevant feature weights to zero.
    - Pros: Balances accuracy and speed. Cons: Model-specific.

**Example**:
- Dataset: 100 features for sales prediction.
- Action: Use correlation to filter low-variance features, then RFE with Random Forest.
- Result: Reduced from 100 to 20 features, improved accuracy by 5%.

**Interview Tips**:
- Relate to curse of dimensionality: “Feature selection avoids overfitting in high-dimensional data.”
- Compare methods: “Filters are fast but naive; wrappers are accurate but slow.”
- Be ready to suggest a method based on dataset size.

---

## 4. How do you handle categorical variables in your dataset?

**Answer**:

Categorical variables (e.g., `color`, `city`) need to be converted into numerical formats for ML models. Common techniques:

- **One-Hot Encoding**:
  - Create binary columns for each category (e.g., `color_red`, `color_blue`).
  - Pros: Works with most algorithms. Cons: High dimensionality for many categories.
  - Example: `color = [red, blue]` → `[1, 0], [0, 1]`.

- **Label Encoding**:
  - Assign integers to categories (e.g., `red=0`, `blue=1`).
  - Pros: Compact. Cons: Implies ordinality, unsuitable for non-ordinal data (e.g., trees).
  - Example: Use for `priority = [low, medium, high]` → `[0, 1, 2]`.

- **Target Encoding**:
  - Replace categories with mean target value (e.g., `city=NY` → avg. sales in NY).
  - Pros: Captures relationship with target. Cons: Risk of data leakage if not careful.
  - Example: For `city`, compute mean `price` per city.

- **Frequency Encoding**:
  - Replace categories with their frequency (e.g., `city=NY` → count of NY).
  - Pros: Simple, low dimensionality. Cons: Loses semantic meaning.

- **Embeddings**:
  - Learn dense vectors for categories (e.g., neural network embeddings).
  - Pros: Captures relationships. Cons: Requires large data, complex models.

**Example**:
- Dataset: `city = [NY, LA, NY, SF]`.
- Action: One-hot encode for Random Forest, target encode for gradient boosting.
- Result: Model handles `city` without ordinality issues.

**Interview Tips**:
- Ask: “Are categories ordinal or nominal?”
- Discuss trade-offs: “One-hot increases dimensions; label encoding risks misinterpretation.”
- Mention model compatibility: “Tree-based models handle label encoding better.”

---

## 5. How do filtering and wrapper methods work in feature selection?

**Answer**:

- **Filter Methods**:
  - **How**: Evaluate features independently of the model using statistical measures.
  - **Steps**:
    1. Compute metric (e.g., correlation, mutual information, chi-squared).
    2. Rank features.
    3. Select top `k` or threshold-based features.
  - **Examples**:
    - Variance: Remove low-variance features.
    - ANOVA: Select features with high class separation.
  - **Pros**: Fast, scalable, model-agnostic.
  - **Cons**: Ignores feature interactions, may miss complex relationships.

- **Wrapper Methods**:
  - **How**: Evaluate feature subsets by training a model and measuring performance.
  - **Steps**:
    1. Start with a subset (e.g., empty or all features).
    2. Add/remove features iteratively (e.g., forward selection, backward elimination).
    3. Evaluate using cross-validation score (e.g., accuracy).
  - **Examples**:
    - Recursive Feature Elimination (RFE): Remove least important features iteratively.
    - Forward Selection: Add features that improve performance.
  - **Pros**: Considers feature interactions, model-specific.
  - **Cons**: Computationally expensive, risk of overfitting to subset.

**Example**:
- Dataset: 50 features for classification.
- Filter: Select top 20 by mutual information (fast).
- Wrapper: Use RFE with SVM to refine to 10 (accurate but slow).

**Interview Tips**:
- Contrast: “Filters are quick but naive; wrappers are thorough but costly.”
- Suggest hybrid: “I’d filter first to reduce features, then use RFE.”
- Be ready to explain a metric (e.g., mutual information measures dependency).

---

## 6. Describe a situation where you had to handle missing data. What techniques did you use?

**Answer**:

**Situation**: I worked on a customer churn prediction project with a dataset containing user demographics and activity logs. About 15% of `income` values and 10% of `last_login` dates were missing.

**Techniques Used**:
1. **Analysis**:
   - Found `income` missingness correlated with `age` (younger users less likely to report).
   - `last_login` missing for inactive users, suggesting a pattern.
2. **Imputation**:
   - **Income**: Imputed with median `income` per `age_group` to preserve distribution.
   - **Last_login**: Imputed with earliest date for inactive users, reflecting their status.
3. **Flagging**:
   - Added `is_income_missing` and `is_login_missing` binary features to capture patterns.
4. **Validation**:
   - Checked imputed values aligned with domain knowledge (e.g., median income reasonable).
   - Compared model performance with/without imputation (F1-score improved with flags).

**Outcome**:
- Model (Random Forest) achieved 0.78 F1-score vs. 0.72 with row deletion.
- Missingness flags improved interpretability (e.g., missing `income` linked to churn).

**Interview Tips**:
- Share a specific story: “Context makes it relatable.”
- Emphasize analysis: “I investigated why data was missing first.”
- Discuss alternatives: “I considered KNN imputation but chose median for simplicity.”

---

## 7. What is principal component analysis (PCA) and when is it used?

**Answer**:

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving most of the variance.

- **How It Works**:
  1. Standardize features (zero mean, unit variance).
  2. Compute covariance matrix to find feature correlations.
  3. Perform eigenvalue decomposition to get principal components (directions of max variance).
  4. Project data onto top `k` components (new features).

- **When Used**:
  - **High-Dimensional Data**: Reduce features (e.g., 1000 to 50) to avoid curse of dimensionality.
  - **Visualization**: Project to 2D/3D for plotting (e.g., customer data).
  - **Noise Reduction**: Remove low-variance components (noise).
  - **Speed**: Lower dimensions speed up training (e.g., image classification).

- **Limitations**:
  - Loses interpretability (components are linear combinations).
  - Assumes linear relationships.
  - Sensitive to scaling.

**Example**:
- Dataset: 1000 pixel features for images.
- Action: Apply PCA, keep 95% variance (reduces to 50 components).
- Result: Faster training, similar accuracy.

**Interview Tips**:
- Explain intuition: “PCA finds directions where data varies most.”
- Mention preprocessing: “Standardization is critical for PCA.”
- Discuss trade-offs: “Reduces dimensions but sacrifices interpretability.”

---

## 8. What's the difference between PCA vs ICA?

**Answer**:

- **PCA (Principal Component Analysis)**:
  - **Goal**: Maximize variance, reduce dimensionality.
  - **How**: Finds orthogonal components (linear combinations) that capture maximum variance.
  - **Assumption**: Data variance is informative (correlated features).
  - **Use Case**:
    - Compress data (e.g., images).
    - Visualize high-dimensional data.
  - **Output**: Components are ordered by variance, uncorrelated.

- **ICA (Independent Component Analysis)**:
  - **Goal**: Separate mixed signals into independent sources.
  - **How**: Finds components that are statistically independent (not just uncorrelated).
  - **Assumption**: Data is a mix of independent sources (e.g., audio signals).
  - **Use Case**:
    - Blind source separation (e.g., separate voices in audio).
    - EEG signal analysis.
  - **Output**: Components are independent, not necessarily orthogonal.

- **Key Differences**:
  - **Objective**: PCA seeks variance; ICA seeks independence.
  - **Output**: PCA components are orthogonal; ICA components are independent.
  - **Application**: PCA for compression; ICA for signal separation.
  - **Assumption**: PCA assumes Gaussian-like variance; ICA assumes non-Gaussian sources.

**Example**:
- PCA: Reduce 100 image features to 10 for classification.
- ICA: Separate two speakers’ voices from a single microphone recording.

**Interview Tips**:
- Clarify goals: “PCA compresses, ICA separates.”
- Use analogies: “PCA is like summarizing a book; ICA is like untangling voices.”
- Be ready to sketch PCA’s variance maximization vs. ICA’s independence.

---

## 9. How do you handle time-based features in a machine learning model?

**Answer**:

Time-based features capture temporal patterns and require careful handling to align with model assumptions.

- **Techniques**:
  - **Extract Components**:
    - Year, month, day, hour, minute, weekday/weekend.
    - Example: From `2023-04-12 15:30`, extract `hour=15`, `weekday=Wednesday`.
  - **Cyclic Encoding**:
    - For periodic features (e.g., hour, month), use sine/cosine transformations to capture cyclical nature.
    - Example: `sin(2π * hour/24)`, `cos(2π * hour/24)`.
  - **Lags and Differences**:
    - Create lag features (e.g., sales at `t-1`, `t-2`).
    - Compute differences (e.g., `sales_t - sales_t-1`).
  - **Rolling Statistics**:
    - Compute moving averages, std, min/max over a window (e.g., last 7 days’ sales).
  - **Time Since Event**:
    - Calculate time since last action (e.g., days since last login).
  - **Seasonality**:
    - Add flags for seasons, holidays, or events.

- **Preprocessing**:
  - Ensure consistent formats (e.g., ISO 8601).
  - Handle missing timestamps (e.g., impute with median).
  - Scale numerical time features (e.g., normalize lags).

**Example**:
- Dataset: Sales data with `timestamp`.
- Features: `hour_sin`, `hour_cos`, `sales_lag1`, `7day_avg_sales`, `is_holiday`.
- Result: Model captures daily and seasonal patterns, improving accuracy.

**Interview Tips**:
- Emphasize cyclic encoding: “Sine/cosine avoids discontinuities in hours.”
- Discuss model fit: “Tree-based models handle raw timestamps; linear models need encoding.”
- Mention leakage: “I’d ensure lags don’t include future data.”

---

## 10. What is feature hashing? When would you use it?

**Answer**:

**Feature hashing** (or hashing trick) maps categorical variables to a fixed-size numerical vector using a hash function, avoiding explicit encoding.

- **How It Works**:
  - Hash each category to an index in a vector of size `m` (e.g., `m=1000`).
  - Increment the index’s value (or use sign for collisions).
  - Example: `city=NY` → hash(NY) % 1000 = 123 → set vector[123] = 1.

- **When to Use**:
  - **High Cardinality**: Many categories (e.g., millions of user IDs).
  - **Memory Constraints**: One-hot encoding too large (e.g., 1M categories → 1M columns).
  - **Streaming Data**: New categories appear during inference.
  - **Quick Prototyping**: Need fast feature representation.

- **Pros**:
  - Fixed memory usage (vector size `m`).
  - Handles new categories without retraining.
  - Fast computation.

- **Cons**:
  - Collisions: Different categories may hash to same index, losing information.
  - Not interpretable (unlike one-hot).
  - Requires tuning `m` (trade-off between collisions and memory).

**Example**:
- Dataset: 1M unique `user_id`s.
- Action: Hash to 10,000-dim vector.
- Result: Fits in memory, model trains faster than one-hot.

**Interview Tips**:
- Explain collisions: “Hashing risks overlap but is fine for high cardinality.”
- Compare with one-hot: “Feature hashing is compact but less precise.”
- Suggest use case: “Ideal for large-scale text or user data.”

---

## 11. How do you handle hierarchical categorical variables?

**Answer**:

Hierarchical categorical variables have a tree-like structure (e.g., `Country > State > City`). Handling them preserves relationships.

- **Techniques**:
  - **Multi-Level Encoding**:
    - Encode each level separately (e.g., one-hot for `Country`, `State`, `City`).
    - Pros: Captures all levels. Cons: High dimensionality.
  - **Target Encoding by Level**:
    - Replace each level with mean target (e.g., `City=NY` → avg. sales in NY, `State=NY` → avg. in NY state).
    - Pros: Reduces dimensions. Cons: Risk of leakage.
  - **Embeddings**:
    - Learn dense vectors capturing hierarchy (e.g., neural network with shared layers for levels).
    - Pros: Models relationships. Cons: Needs large data.
  - **Path Encoding**:
    - Treat hierarchy as a single feature (e.g., `US/NY/NYC`).
    - Pros: Compact. Cons: Loses granularity unless paired with other encodings.
  - **Group Aggregates**:
    - Add features like avg. target per `State` or `Country`.
    - Pros: Captures broader trends.

**Example**:
- Dataset: `Country=US, State=NY, City=NYC`.
- Action: One-hot encode `City`, target encode `State`, add `Country` avg. sales.
- Result: Model learns local and regional patterns.

**Interview Tips**:
- Clarify hierarchy depth: “How many levels are there?”
- Discuss trade-offs: “Multi-level encoding is precise but bulky.”
- Mention tree-based models: “They handle hierarchies well with raw labels.”

---

## 12. What are embedding layers and when should you use them?

**Answer**:

**Embedding layers** are neural network layers that map categorical variables to dense, low-dimensional vectors, learning meaningful representations.

- **How They Work**:
  - Each category (e.g., `word=cat`) is assigned a vector (e.g., `[0.2, -0.1, 0.5]`).
  - Vectors are learned during training to capture relationships (e.g., `cat` and `dog` vectors are similar).
  - Output: Fixed-size vector per category (e.g., 50 dimensions).

- **When to Use**:
  - **High Cardinality**: Many categories (e.g., words, users).
  - **Complex Relationships**: Categories have semantic similarity (e.g., `king` vs. `queen`).
  - **Neural Networks**: Embedding layers integrate naturally with deep learning.
  - **Large Data**: Enough data to learn robust embeddings.
  - **NLP/Vision**: Pretrained embeddings (e.g., Word2Vec, BERT) for text or images.

- **Pros**:
  - Captures relationships (unlike one-hot).
  - Low-dimensional, memory-efficient.
  - Generalizes to new categories (with pretrained embeddings).
- **Cons**:
  - Requires large data to train.
  - Less interpretable than one-hot.
  - Computationally expensive.

**Example**:
- Task: Sentiment analysis on reviews.
- Action: Use embedding layer for words (50-dim vectors).
- Result: Model learns `good` and `great` are similar, improving accuracy.

**Interview Tips**:
- Contrast with one-hot: “Embeddings are compact and learn relationships.”
- Mention pretrained: “For small data, I’d use GloVe or BERT embeddings.”
- Explain training: “Vectors adjust via backpropagation.”

---

## 13. Explain different strategies for handling outliers in different ML algorithms

**Answer**:

Outliers can skew model performance, but handling depends on the algorithm.

- **Linear Models (e.g., Linear Regression, Logistic Regression)**:
  - **Impact**: Sensitive to outliers (affect coefficients via squared loss).
  - **Strategies**:
    - **Remove**: Clip outliers (e.g., cap `income` at 99th percentile).
    - **Transform**: Apply log or robust scaling (e.g., `log(income)`).
    - **Robust Loss**: Use Huber loss (less sensitive to outliers).
  - **Example**: Cap extreme `price` values to stabilize regression.

- **Tree-Based Models (e.g., Random Forest, XGBoost)**:
  - **Impact**: Robust to outliers (splits based on order, not magnitude).
  - **Strategies**:
    - **Minimal Handling**: Often ignore outliers unless they’re errors.
    - **Feature Engineering**: Add binary flag for outliers (e.g., `is_high_price`).
  - **Example**: Keep outliers, as trees handle them naturally.

- **Distance-Based Models (e.g., K-Means, KNN)**:
  - **Impact**: Sensitive (outliers distort distances).
  - **Strategies**:
    - **Remove**: Filter outliers using IQR (e.g., outside 1.5 * IQR).
    - **Transform**: Standardize or log-transform features.
    - **Robust Algorithms**: Use DBSCAN (treats outliers as noise).
  - **Example**: Remove extreme `distance` values for K-Means clustering.

- **Neural Networks**:
  - **Impact**: Sensitive (outliers affect gradients).
  - **Strategies**:
    - **Clip**: Bound inputs (e.g., clip pixel values).
    - **Normalize**: Scale to [0, 1] or standardize.
    - **Robust Activation**: Use bounded functions (e.g., sigmoid).
  - **Example**: Standardize image pixels to handle bright spots.

- **General Detection**:
  - **IQR**: Flag values outside Q1 - 1.5*IQR or Q3 + 1.5*IQR.
  - **Z-Score**: Flag values with `|z| > 3`.
  - **Domain Knowledge**: Define outliers (e.g., `age > 120` is invalid).

**Example**:
- Dataset: Sales with outlier `revenue = $1M` (avg. $10K).
- Linear Regression: Clip at 99th percentile.
- Random Forest: Keep as is.
- K-Means: Remove via IQR.

**Interview Tips**:
- Ask: “Are outliers meaningful or errors?” (e.g., fraud vs. typo).
- Tailor to algorithm: “Trees are robust, but linear models need clipping.”
- Emphasize detection: “I’d use IQR or domain rules to identify outliers.”

---

## Notes

- **Practicality**: Answers provide actionable techniques with pros/cons, grounded in real-world scenarios.
- **Clarity**: Explanations are structured for verbal delivery, with examples and trade-offs.
- **Depth**: Balances technical detail (e.g., PCA math, embedding training) with accessibility for interview settings.
- **Consistency**: Matches the style of previous files for a cohesive repository.

For deeper practice, try implementing feature engineering (see [ML Coding](ml-coding.md)) or explore [ML System Design](ml-system-design.md) for scaling data pipelines. 🚀

---

**Next Steps**: Strengthen your skills with [Tree-Based Models](tree-based-models.md) for algorithm-specific questions or revisit [Easy ML](easy-ml.md) for fundamentals! 🌟