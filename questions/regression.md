# Regression Questions

This file contains regression-related questions commonly asked in interviews at companies like **Google**, **Amazon**, and others. These questions assess your **understanding** of regression techniques, their assumptions, evaluation metrics, and practical applications. They test your ability to articulate regression concepts and apply them to real-world problems.

Below are the questions with detailed answers, including explanations, mathematical intuition where relevant, and practical insights for interviews.

---

## Table of Contents

1. [What is regression analysis?](#1-what-is-regression-analysis)
2. [What is the difference between linear regression and logistic regression?](#2-what-is-the-difference-between-linear-regression-and-logistic-regression)
3. [What are the assumptions of linear regression?](#3-what-are-the-assumptions-of-linear-regression)
4. [How do you evaluate the performance of a regression model?](#4-how-do-you-evaluate-the-performance-of-a-regression-model)
5. [What is multicollinearity and how do you detect it?](#5-what-is-multicollinearity-and-how-do-you-detect-it)
6. [What is the difference between Ridge and Lasso regression?](#6-what-is-the-difference-between-ridge-and-lasso-regression)
7. [What is polynomial regression and when would you use it?](#7-what-is-polynomial-regression-and-when-would-you-use-it)
8. [How does quantile regression differ from ordinary least squares regression?](#8-how-does-quantile-regression-differ-from-ordinary-least-squares-regression)

---

## 1. What is regression analysis?

**Answer**:

**Regression analysis** is a statistical method used to model the relationship between a dependent variable (target) and one or more independent variables (features) to predict continuous outcomes.

- **Key Points**:
  - **Goal**: Estimate how features influence the target (e.g., predict house prices from size, location).
  - **Types**:
    - **Linear**: Assumes linear relationship (e.g., `y = w0 + w1*x1`).
    - **Non-Linear**: Captures complex patterns (e.g., polynomial regression).
    - **Logistic**: For binary outcomes (not strictly regression, but related).
  - **Components**:
    - Model (e.g., linear equation).
    - Loss function (e.g., Mean Squared Error).
    - Optimization (e.g., least squares).

- **Use Cases**:
  - Predict sales, stock prices, temperature.
  - Understand feature impact (e.g., “How does age affect income?”).

**Example**:
- Task: Predict house price from square footage.
- Model: `price = 50,000 + 200 * sqft`.
- Result: Predicts price for new houses.

**Interview Tips**:
- Clarify scope: “Focuses on continuous outputs, unlike classification.”
- Mention flexibility: “Can be linear or non-linear.”
- Be ready to sketch: “Show y vs. x with a fitted line.”

---

## 2. What is the difference between linear regression and logistic regression?

**Answer**:

- **Linear Regression**:
  - **Purpose**: Predicts a continuous outcome.
  - **Model**: `y = w0 + w1*x1 + ... + wn*xn`.
  - **Output**: Unbounded real numbers (e.g., price, temperature).
  - **Loss**: Mean Squared Error (MSE): `1/n * Σ(y_pred - y_true)²`.
  - **Use Case**: Predict house prices, sales.
  - **Assumption**: Linear relationship, Gaussian errors.

- **Logistic Regression**:
  - **Purpose**: Predicts probability of a binary outcome (extendable to multiclass).
  - **Model**: `P(y=1) = 1/(1 + e^-(w0 + w1*x1 + ...))` (sigmoid function).
  - **Output**: Probability [0, 1], thresholded for class (e.g., 0.5).
  - **Loss**: Log loss (cross-entropy): `-1/n * Σ[y_true * log(y_pred) + (1-y_true) * log(1-y_pred)]`.
  - **Use Case**: Predict churn (yes/no), spam detection.
  - **Assumption**: Log-odds are linear.

- **Key Differences**:
  - **Output**: Linear → continuous; Logistic → probability.
  - **Task**: Linear for regression; Logistic for classification.
  - **Loss**: Linear uses MSE; Logistic uses log loss.
  - **Non-Linearity**: Logistic applies sigmoid to linear combination.

**Example**:
- Linear: Predict weight from height (e.g., 70 kg).
- Logistic: Predict if someone exercises (yes/no) based on age.

**Interview Tips**:
- Emphasize output: “Logistic outputs probabilities, not raw values.”
- Clarify naming: “Logistic is classification despite ‘regression’ name.”
- Be ready to derive: “Show sigmoid transforming linear model.”

---

## 3. What are the assumptions of linear regression?

**Answer**:

Linear regression relies on several assumptions for valid results:

1. **Linearity**:
   - The relationship between features and target is linear (`y = w0 + w1*x1 + ...`).
   - Check: Scatter plots, residual plots (no patterns).
2. **Independence**:
   - Observations are independent of each other.
   - Check: Study design (e.g., no repeated measures).
3. **Homoscedasticity**:
   - Constant variance of residuals across feature values.
   - Check: Residual vs. fitted plot (even spread).
4. **Normality**:
   - Residuals are normally distributed (for inference, not prediction).
   - Check: Q-Q plot, Shapiro-Wilk test.
5. **No Multicollinearity**:
   - Features are not highly correlated with each other.
   - Check: Variance Inflation Factor (VIF), correlation matrix.
6. **No Extreme Outliers**:
   - Outliers can skew coefficients.
   - Check: Boxplots, leverage statistics.

**Example**:
- Task: Predict sales from ad spend.
- Violation: Non-linear pattern in residuals → use polynomial regression.

**Interview Tips**:
- Prioritize key assumptions: “Linearity and homoscedasticity are critical.”
- Mention checks: “I’d plot residuals to verify.”
- Discuss fixes: “Transform features for non-linearity.”

---

## 4. How do you evaluate the performance of a regression model?

**Answer**:

Evaluating a regression model involves metrics that measure prediction error and fit quality:

- **Mean Squared Error (MSE)**:
  - `1/n * Σ(y_pred - y_true)²`.
  - Pros: Penalizes large errors, differentiable.
  - Cons: Sensitive to outliers, scale-dependent.
- **Root Mean Squared Error (RMSE)**:
  - `√MSE`.
  - Pros: Interpretable in target units (e.g., dollars).
  - Cons: Still outlier-sensitive.
- **Mean Absolute Error (MAE)**:
  - `1/n * Σ|y_pred - y_true|`.
  - Pros: Robust to outliers, intuitive.
  - Cons: Less sensitive to large errors.
- **R-Squared (R²)**:
  - `1 - (Σ(y_pred - y_true)² / Σ(y_true - mean(y_true))²)`.
  - Pros: Measures variance explained (0 to 1).
  - Cons: Can mislead with non-linear models.
- **Adjusted R-Squared**:
  - Adjusts R² for number of features.
  - Pros: Penalizes overfitting.
- **Residual Analysis**:
  - Plot residuals vs. predicted or features.
  - Pros: Diagnoses assumption violations (e.g., non-linearity).

**Example**:
- Model: Predict house prices.
- Metrics: RMSE = $10,000, R² = 0.85.
- Residuals: Random scatter → good fit.

**Interview Tips**:
- Tailor metrics: “RMSE for interpretability, MAE for robustness.”
- Discuss context: “Business prefers dollar errors (RMSE).”
- Be ready to plot: “Show residuals to check fit.”

---

## 5. What is multicollinearity and how do you detect it?

**Answer**:

**Multicollinearity** occurs when independent variables in a regression model are highly correlated, leading to unstable or misleading coefficients.

- **Impact**:
  - Inflates standard errors, making coefficients insignificant.
  - Hard to interpret feature importance (e.g., which variable drives `y`?).
  - Does not affect predictions, only inference.

- **Detection**:
  - **Correlation Matrix**:
    - Compute Pearson correlation between features.
    - Threshold: `|r| > 0.8` suggests issue.
  - **Variance Inflation Factor (VIF)**:
    - `VIF_i = 1/(1 - R²_i)`, where `R²_i` is from regressing feature `i` on others.
    - Threshold: VIF > 5 or 10 indicates multicollinearity.
  - **Condition Number**:
    - Ratio of largest to smallest eigenvalue of feature matrix.
    - High value (>30) suggests instability.

- **Solutions**:
  - Remove one correlated feature.
  - Combine features (e.g., PCA, average).
  - Use regularized models (Ridge, Lasso).

**Example**:
- Features: `height_cm`, `height_m` (r = 1.0).
- VIF: >1000.
- Action: Drop `height_m`.

**Interview Tips**:
- Clarify impact: “Affects inference, not predictions.”
- Suggest VIF: “Most reliable for detection.”
- Be ready to compute: “Show VIF formula.”

---

## 6. What is the difference between Ridge and Lasso regression?

**Answer**:

- **Ridge Regression**:
  - **How**: Adds L2 penalty to linear regression loss: `MSE + λ * Σw_i²`.
  - **Effect**: Shrinks coefficients toward zero, but rarely to zero.
  - **Use Case**: Handles multicollinearity, stabilizes coefficients.
  - **Pros**: Robust to correlated features.
  - **Cons**: Keeps all features (less interpretable).

- **Lasso Regression**:
  - **How**: Adds L1 penalty: `MSE + λ * Σ|w_i|`.
  - **Effect**: Shrinks coefficients, sets some to exactly zero (feature selection).
  - **Use Case**: Sparse models, high-dimensional data.
  - **Pros**: Selects features automatically.
  - **Cons**: Unstable with highly correlated features.

- **Key Differences**:
  - **Penalty**: Ridge uses L2 (squared); Lasso uses L1 (absolute).
  - **Feature Selection**: Lasso eliminates features; Ridge shrinks them.
  - **Stability**: Ridge better for multicollinearity; Lasso may pick one of correlated pair.
  - **Geometry**: Ridge constrains weights to a sphere; Lasso to a diamond.

**Example**:
- Dataset: 100 features, some correlated.
- Ridge: Keeps all features, small weights.
- Lasso: Selects 20 features, others zero.

**Interview Tips**:
- Explain penalty: “L1 promotes sparsity, L2 smoothness.”
- Mention Elastic Net: “Combines both for balance.”
- Be ready to sketch: “Show L1 vs. L2 constraint shapes.”

---

## 7. What is polynomial regression and when would you use it?

**Answer**:

**Polynomial regression** extends linear regression by modeling non-linear relationships using polynomial terms of features.

- **How It Works**:
  - Instead of `y = w0 + w1*x`, use `y = w0 + w1*x + w2*x² + ... + wn*x^n`.
  - Fit using least squares, treating `x²`, `x³` as new features.
  - Can include interactions (e.g., `x1 * x2`).

- **When to Use**:
  - **Non-Linear Data**: When scatter plots show curvature (e.g., sales vs. time).
  - **Simple Non-Linearity**: Polynomial terms suffice (vs. complex models like neural nets).
  - **Interpretability**: Need to explain relationship (vs. black-box models).
  - **Small Datasets**: Avoids overfitting of deep models.

- **Limitations**:
  - **Overfitting**: High-degree polynomials fit noise (e.g., degree > 5).
  - **Extrapolation**: Poor outside training range.
  - **Computation**: High-degree terms increase complexity.

**Example**:
- Task: Predict car speed vs. time (curved).
- Model: `speed = w0 + w1*time + w2*time²`.
- Result: Captures acceleration curve.

**Interview Tips**:
- Clarify fit: “It’s still linear in weights, just non-linear in features.”
- Discuss degree: “Choose via cross-validation to avoid overfitting.”
- Be ready to plot: “Show linear vs. polynomial fit.”

---

## 8. How does quantile regression differ from ordinary least squares regression?

**Answer**:

- **Ordinary Least Squares (OLS) Regression**:
  - **Goal**: Minimize mean squared error to predict the mean of the target.
  - **Loss**: `1/n * Σ(y_pred - y_true)²`.
  - **Output**: Conditional mean (`E[y|x]`).
  - **Assumption**: Homoscedastic errors, focuses on central tendency.
  - **Use Case**: Predict average house price.

- **Quantile Regression**:
  - **Goal**: Predict specific quantiles of the target (e.g., median, 90th percentile).
  - **Loss**: Weighted absolute error: `Σρ_τ(y_pred - y_true)`, where `ρ_τ(u) = u * (τ - I(u<0))`, `τ` is quantile (e.g., 0.5 for median).
  - **Output**: Conditional quantile (e.g., `Q_τ(y|x)`).
  - **Assumption**: No assumption on error distribution, robust to heteroscedasticity.
  - **Use Case**: Predict price ranges, model tails (e.g., high-risk cases).

- **Key Differences**:
  - **Focus**: OLS predicts mean; quantile predicts quantiles.
  - **Loss**: OLS uses squared error; quantile uses asymmetric absolute error.
  - **Robustness**: Quantile is robust to outliers, heteroscedasticity.
  - **Interpretation**: Quantile models entire distribution (e.g., median vs. 95th percentile).

**Example**:
- Task: Predict income.
- OLS: Average income = $50K.
- Quantile (τ=0.9): 90th percentile = $100K.

**Interview Tips**:
- Explain quantiles: “Captures distribution, not just mean.”
- Highlight robustness: “Great for skewed or outlier-heavy data.”
- Be ready to derive: “Show quantile loss function.”

---

## Notes

- **Focus**: Answers emphasize regression-specific concepts, ideal for ML interviews.
- **Clarity**: Explanations are structured for verbal delivery, with examples and trade-offs.
- **Depth**: Includes mathematical intuition (e.g., loss functions) and practical tips (e.g., VIF for multicollinearity).
- **Consistency**: Matches the style of previous files for a cohesive repository.

For deeper practice, try implementing regression models (see [ML Coding](ml-coding.md)) or explore [ML System Design](ml-system-design.md) for scaling regression solutions. 🚀

---

**Next Steps**: Build on these skills with [Statistics & Probability](statistics-probability.md) for foundational math or revisit [Deep Learning](deep-learning.md) for advanced regression models! 🌟