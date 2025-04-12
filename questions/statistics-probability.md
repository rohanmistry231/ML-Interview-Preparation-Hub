# Statistics and Probability Questions

This file contains statistics and probability questions commonly asked in machine learning interviews at companies like **Google**, **Amazon**, **Meta**, and others. These questions assess your **foundational understanding** of statistical concepts, probability theory, and their applications in ML, such as model evaluation, uncertainty quantification, and data analysis.

Below are the questions with detailed answers, including explanations, mathematical intuition, and practical insights for interviews.

---

## Table of Contents

1. [What is the difference between population and sample?](#1-what-is-the-difference-between-population-and-sample)
2. [Explain the Central Limit Theorem and its significance in machine learning](#2-explain-the-central-limit-theorem-and-its-significance-in-machine-learning)
3. [What is a p-value, and how is it used in hypothesis testing?](#3-what-is-a-p-value-and-how-is-it-used-in-hypothesis-testing)
4. [What is the difference between Type I and Type II errors?](#4-what-is-the-difference-between-type-i-and-type-ii-errors)
5. [What are bias and variance in the context of machine learning?](#5-what-are-bias-and-variance-in-the-context-of-machine-learning)
6. [What is the difference between a probability density function and a cumulative distribution function?](#6-what-is-the-difference-between-a-probability-density-function-and-a-cumulative-distribution-function)
7. [Explain Bayes’ Theorem and its applications in machine learning](#7-explain-bayes-theorem-and-its-applications-in-machine-learning)
8. [What is the difference between correlation and causation?](#8-what-is-the-difference-between-correlation-and-causation)

---

## 1. What is the difference between population and sample?

**Answer**:

- **Population**:
  - **Definition**: The entire set of individuals or observations that the study aims to describe (e.g., all users of an app).
  - **Parameters**: Described by true values (e.g., population mean `μ`, variance `σ²`).
  - **Use**: Ideal for analysis but often impractical due to size or cost.
  - **Example**: All global smartphone users.

- **Sample**:
  - **Definition**: A subset of the population selected for analysis (e.g., 1000 surveyed users).
  - **Statistics**: Estimates population parameters (e.g., sample mean `x̄`, variance `s²`).
  - **Use**: Practical for inference, assuming proper sampling (e.g., random).
  - **Example**: 1000 randomly selected smartphone users.

- **Key Differences**:
  - **Scope**: Population is complete; sample is partial.
  - **Values**: Population has true parameters; sample has estimates.
  - **Purpose**: Population defines truth; sample approximates it.
  - **ML Context**: Models train on samples (e.g., datasets) to generalize to populations.

**Example**:
- Population: All customer transactions.
- Sample: 10,000 transactions for churn prediction model.

**Interview Tips**:
- Clarify inference: “Samples estimate population traits.”
- Mention sampling: “Random sampling reduces bias.”
- Be ready to discuss: “Impact of bad sampling on ML.”

---

## 2. Explain the Central Limit Theorem and its significance in machine learning

**Answer**:

The **Central Limit Theorem (CLT)** states that the distribution of the sample mean (or sum) of a sufficiently large number of independent, identically distributed (i.i.d.) random variables approaches a normal distribution, regardless of the underlying distribution, provided the variance is finite.

- **Key Points**:
  - **Conditions**:
    - Sample size `n` is large (typically `n ≥ 30`).
    - Variables are i.i.d.
    - Finite mean `μ` and variance `σ²`.
  - **Result**: Sample mean `x̄ ~ N(μ, σ²/n)` (standard error = `σ/√n`).
  - **Intuition**: Averages smooth out quirks of the original distribution.

- **Significance in ML**:
  - **Confidence Intervals**: Estimate model metrics (e.g., accuracy) with normal-based intervals.
  - **Hypothesis Testing**: Use normal assumptions for p-values in A/B tests.
  - **Data Analysis**: Justifies normality of aggregated metrics (e.g., average user time).
  - **Feature Engineering**: Supports scaling/transformations assuming normality.
  - **Gradient Descent**: Errors in large batches approximate normality, aiding optimization.

**Example**:
- Data: User session times (skewed).
- CLT: Mean of 100 sessions ~ normal, enabling t-tests for significance.

**Interview Tips**:
- Emphasize normality: “CLT makes means normal for large `n`.”
- Link to ML: “Critical for testing model performance.”
- Be ready to sketch: “Show skewed data → normal means.”

---

## 3. What is a p-value, and how is it used in hypothesis testing?

**Answer**:

A **p-value** is the probability of observing data (or more extreme) under the null hypothesis, used to assess evidence against it.

- **How It Works**:
  - **Null Hypothesis (H₀)**: Assumes no effect (e.g., “new model = old model”).
  - **Alternative Hypothesis (H₁)**: Claims an effect (e.g., “new model > old”).
  - **Test Statistic**: Compute metric (e.g., t-statistic) from data.
  - **P-value**: `P(data | H₀)`—small p-value suggests H₀ is unlikely.
  - **Threshold (α)**: Reject H₀ if p < α (e.g., 0.05).

- **In Hypothesis Testing**:
  - **Steps**:
    1. Define H₀, H₁.
    2. Choose test (e.g., t-test, chi-squared).
    3. Compute p-value.
    4. Decide: Reject H₀ if p < α, else fail to reject.
  - **Interpretation**: Small p-value indicates strong evidence against H₀.

- **ML Use Cases**:
  - A/B testing: Compare model performance (e.g., click rates).
  - Feature selection: Test feature significance.
  - Model evaluation: Check if improvements are significant.

**Example**:
- Test: Does new model increase accuracy?
- H₀: No difference.
- P-value = 0.03 < 0.05 → Reject H₀, new model is better.

**Interview Tips**:
- Clarify meaning: “P-value measures evidence, not truth.”
- Avoid pitfalls: “Small p doesn’t prove H₁.”
- Be ready to compute: “Explain t-test p-value.”

---

## 4. What is the difference between Type I and Type II errors?

**Answer**:

- **Type I Error (False Positive)**:
  - **Definition**: Reject the null hypothesis (H₀) when it is true.
  - **Probability**: Denoted `α` (significance level, e.g., 0.05).
  - **Example**: Conclude a model improves accuracy (H₀: no improvement) when it doesn’t.
  - **Impact**: Overconfidence in results, deploying ineffective models.
  - **ML Context**: False alarm in fraud detection (flag innocent transaction).

- **Type II Error (False Negative)**:
  - **Definition**: Fail to reject H₀ when it is false.
  - **Probability**: Denoted `β` (1 - power).
  - **Example**: Miss that a model improves accuracy (H₀ false) when it does.
  - **Impact**: Missed opportunities, underutilizing better models.
  - **ML Context**: Miss fraud in detection (let guilty transaction pass).

- **Key Differences**:
  - **Error Type**: Type I = wrong rejection; Type II = wrong acceptance.
  - **Probability**: Type I controlled by `α`; Type II by `β`.
  - **Trade-Off**: Reducing Type I (lower `α`) increases Type II, and vice versa.
  - **Priority**: Depends on context (e.g., medical tests prioritize low Type II).

**Example**:
- Fraud Model:
  - Type I: Flag innocent user (H₀: not fraud, rejected).
  - Type II: Miss fraudster (H₀: not fraud, accepted).

**Interview Tips**:
- Use terms: “Type I is false positive, Type II is false negative.”
- Discuss trade-offs: “Lower α reduces Type I but risks Type II.”
- Be ready to contextualize: “Fraud needs low Type II.”

---

## 5. What are bias and variance in the context of machine learning?

**Answer**:

- **Bias**:
  - **Definition**: Error due to overly simplistic models (underfitting).
  - **Cause**: Model assumes too much (e.g., linear model for non-linear data).
  - **Effect**: High training error, poor fit to data.
  - **Example**: Linear regression on quadratic data misses curves.
  - **Math**: `Bias = E[f̂(x)] - f(x)`, where `f̂` is model, `f` is truth.

- **Variance**:
  - **Definition**: Error due to sensitivity to training data (overfitting).
  - **Cause**: Model too complex (e.g., deep tree fits noise).
  - **Effect**: Low training error, high test error.
  - **Example**: Decision tree memorizes training points, fails on new data.
  - **Math**: `Variance = E[(f̂(x) - E[f̂(x)])²]`.

- **Bias-Variance Trade-Off**:
  - **Goal**: Minimize total error = `Bias² + Variance + Irreducible Error`.
  - **High Bias**: Simple models (e.g., linear regression).
  - **High Variance**: Complex models (e.g., deep nets).
  - **Balance**: Use regularization, ensembles, or model selection.

**Example**:
- Task: Predict house prices.
- High Bias: Linear model (misses patterns, error=10K).
- High Variance: 100-layer net (fits noise, test error=15K).
- Balanced: Ridge regression (error=5K).

**Interview Tips**:
- Explain intuition: “Bias underfits, variance overfits.”
- Mention trade-off: “Tune complexity for balance.”
- Be ready to sketch: “Show error vs. complexity curve.”

---

## 6. What is the difference between a probability density function and a cumulative distribution function?

**Answer**:

- **Probability Density Function (PDF)**:
  - **Definition**: For continuous random variables, describes the likelihood of a variable taking a specific value.
  - **Properties**:
    - `f(x) ≥ 0` (non-negative).
    - Integral over all `x`: `∫f(x)dx = 1`.
    - Probability over interval: `P(a ≤ X ≤ b) = ∫_a^b f(x)dx`.
  - **Use**: Model distributions (e.g., Gaussian for errors).
  - **Example**: Normal PDF `f(x) = (1/√(2πσ²))e^(-(x-μ)²/(2σ²))`.

- **Cumulative Distribution Function (CDF)**:
  - **Definition**: Gives the probability that a random variable is less than or equal to a value: `F(x) = P(X ≤ x)`.
  - **Properties**:
    - `0 ≤ F(x) ≤ 1`.
    - Monotonically increasing.
    - `F(x) = ∫_{-∞}^x f(t)dt` (for continuous variables).
  - **Use**: Compute probabilities, quantiles (e.g., 95th percentile).
  - **Example**: Normal CDF gives `P(X ≤ 1)`.

- **Key Differences**:
  - **Role**: PDF gives density at a point; CDF gives cumulative probability.
  - **Output**: PDF can exceed 1 (density); CDF is [0,1].
  - **Computation**: PDF is derivative of CDF; CDF is integral of PDF.
  - **ML Context**: PDF for likelihoods (e.g., GMM); CDF for thresholds (e.g., anomaly detection).

**Example**:
- Normal Distribution:
  - PDF: Peak at mean, describes spread.
  - CDF: Sigmoid-like, gives `P(X < 0)`.

**Interview Tips**:
- Clarify continuous: “PDF for continuous, PMF for discrete.”
- Link to ML: “PDF in loss, CDF in thresholds.”
- Be ready to plot: “Show PDF vs. CDF curves.”

---

## 7. Explain Bayes’ Theorem and its applications in machine learning

**Answer**:

**Bayes’ Theorem** describes how to update probabilities based on new evidence, fundamental to probabilistic reasoning.

- **Formula**:
  - `P(A|B) = [P(B|A) * P(A)] / P(B)`.
  - **Terms**:
    - `P(A|B)`: Posterior (probability of A given B).
    - `P(B|A)`: Likelihood (probability of B given A).
    - `P(A)`: Prior (initial belief about A).
    - `P(B)`: Evidence (normalizing constant).

- **Intuition**:
  - Combines prior knowledge with observed data to refine beliefs.
  - Example: Update disease probability given test result.

- **Applications in ML**:
  - **Naive Bayes Classifier**:
    - Assumes feature independence.
    - Uses Bayes to compute `P(class|features)`.
    - Example: Spam detection.
  - **Bayesian Inference**:
    - Update model parameters (e.g., Gaussian Process priors).
    - Example: Hyperparameter tuning.
  - **Probabilistic Models**:
    - VAEs, Bayesian neural nets model uncertainty.
    - Example: Predict with confidence intervals.
  - **Anomaly Detection**:
    - Compute `P(data|normal)` vs. `P(data|anomaly)`.
  - **Recommendation Systems**:
    - Update user preferences based on interactions.

**Example**:
- Task: Diagnose disease (1% prevalence, test 95% accurate).
- Bayes: `P(disease|positive) = [P(positive|disease) * P(disease)] / P(positive)`.

**Interview Tips**:
- Break down formula: “Prior, likelihood, posterior.”
- Highlight ML: “Naive Bayes is a direct application.”
- Be ready to derive: “Compute spam example.”

---

## 8. What is the difference between correlation and causation?

**Answer**:

- **Correlation**:
  - **Definition**: Measures the strength and direction of a linear relationship between two variables.
  - **Metric**: Pearson correlation coefficient `r` (-1 to 1).
    - `r = cov(X,Y) / (σ_X * σ_Y)`.
  - **Properties**:
    - No implication of cause.
    - Can be spurious (e.g., ice cream sales vs. drownings).
  - **ML Use**: Feature selection, exploratory analysis.
  - **Example**: Height and weight (r = 0.7).

- **Causation**:
  - **Definition**: One variable directly influences another (cause → effect).
  - **Establishing**:
    - Randomized experiments (e.g., RCTs).
    - Causal inference (e.g., propensity scoring, DAGs).
  - **Properties**:
    - Requires control for confounders.
    - Harder to prove than correlation.
  - **ML Use**: Policy decisions, treatment effects.
  - **Example**: Medicine improves health (proven via trial).

- **Key Differences**:
  - **Implication**: Correlation shows association; causation shows effect.
  - **Proof**: Correlation is statistical; causation needs experiments or causal models.
  - **Risk**: Assuming correlation = causation leads to errors.
  - **ML Context**: Correlation for patterns, causation for actions.

**Example**:
- Correlation: More ads, higher sales (r = 0.8).
- Causation: Prove ads drive sales via A/B test.

**Interview Tips**:
- Stress caution: “Correlation doesn’t imply causation.”
- Mention confounders: “Hidden variables cause spurious links.”
- Be ready to clarify: “Use RCTs for causation.”

---

## Notes

- **Focus**: Answers cover core stats and probability, critical for ML interviews.
- **Clarity**: Explanations are structured for verbal delivery, with examples and trade-offs.
- **Depth**: Includes mathematical rigor (e.g., Bayes, CLT) and ML applications (e.g., hypothesis testing in A/B tests).
- **Consistency**: Matches the style of previous files for a cohesive repository.

For deeper practice, apply these concepts in model evaluation (see [ML Algorithms](ml-algorithms.md)) or explore [Advanced ML](advanced-ml.md) for Bayesian methods. 🚀

---

**Next Steps**: Build on these skills with [ML System Design](ml-system-design.md) for scaling statistical models or revisit [Time Series & Clustering](time-series-clustering.md) for time-based stats! 🌟