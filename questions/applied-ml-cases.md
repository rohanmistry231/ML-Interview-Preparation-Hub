# Applied ML Cases Questions

This file contains applied machine learning case questions commonly asked in interviews at companies like **Google**, **PayPal**, and **Apple**. These questions assess your ability to **solve real-world business problems** using machine learning, focusing on translating business needs into ML solutions. They test your practical skills, often requiring verbal explanations or hands-on coding in environments like Jupyter or Colab.

Below are the questions with detailed answers, including step-by-step approaches, key considerations, and practical insights for interviews.

---

## Table of Contents

1. [Given a dataset that contains purchase history on PlayStore, how would you build a propensity score model?](#1-given-a-dataset-that-contains-purchase-history-on-playstore-how-would-you-build-a-propensity-score-model)
2. [How would you build a fraud model without labels?](#2-how-would-you-build-a-fraud-model-without-labels)
3. [How would you identify meaningful segmentation?](#3-how-would-you-identify-meaningful-segmentation)

---

## 1. Given a dataset that contains purchase history on PlayStore, how would you build a propensity score model?

**Question**: [Google] You have a dataset with purchase history on the PlayStore. Describe how you would build a propensity score model to predict the likelihood of a user making a purchase.

**Answer**:

A **propensity score model** predicts the probability that a user will perform a specific action (here, making a purchase). For the PlayStore purchase history dataset, the goal is to estimate the likelihood of a user buying an app or in-app item. Here’s a step-by-step approach:

1. **Problem Definition**:
   - **Objective**: Binary classification to predict `P(purchase = 1 | user features)`.
   - **Output**: Probability score (0 to 1) for each user.
   - **Metric**: AUC-ROC (for ranking users), precision/recall (for business impact).

2. **Data Exploration**:
   - **Dataset**: Assume columns like `user_id`, `app_id`, `timestamp`, `price`, `category`, `past_purchases`, `session_duration`, `device_type`.
   - **Target**: Create a binary label (`1` for purchased, `0` for no purchase within a time window).
   - **EDA**: Check for class imbalance (purchases are rare), missing values, and correlations (e.g., session time vs. purchase).

3. **Feature Engineering**:
   - **User Features**:
     - Frequency of app visits, time since last purchase.
     - Total spend, average purchase value.
     - Session metrics (e.g., avg. session duration, clicks per session).
   - **App Features**:
     - App category (e.g., games, productivity), price, ratings.
   - **Behavioral Features**:
     - Recency, frequency, monetary (RFM) scores.
     - Interaction patterns (e.g., viewed but didn’t buy).
   - **Temporal Features**:
     - Day of week, time of day for purchases.
   - Handle categorical variables (e.g., one-hot encode `category`, `device_type`).

4. **Data Preprocessing**:
   - **Imbalanced Data**: Use oversampling (SMOTE), undersampling, or class weights due to rare purchases.
   - **Missing Values**: Impute numerical features (e.g., median for session time) and categorical features (e.g., mode for device).
   - **Scaling**: Standardize numerical features (e.g., spend, session duration) for models like logistic regression.

5. **Model Selection**:
   - **Algorithm**: Start with **logistic regression** (interpretable, outputs probabilities) or **Random Forest/XGBoost** (handles non-linearity, interactions).
   - **Why Logistic Regression**: Easy to interpret coefficients (e.g., “+10 min session time increases purchase odds by X%”).
   - **Why XGBoost**: Captures complex patterns, often performs well for imbalanced data.

6. **Training and Evaluation**:
   - **Split**: Use train-validation-test split (e.g., 70-15-15) or time-based split (train on older data, test on recent).
   - **Cross-Validation**: 5-fold CV to tune hyperparameters (e.g., regularization for logistic regression, `max_depth` for XGBoost).
   - **Metrics**:
     - **Primary**: AUC-ROC to evaluate ranking ability.
     - **Secondary**: Precision/recall/F1 for minority class (purchases).
     - **Business**: Expected revenue from targeting top N% of users.

7. **Model Interpretation**:
   - Use feature importance (e.g., XGBoost’s gain) or SHAP values to identify key drivers (e.g., high spend, frequent sessions).
   - Explain to stakeholders: “Users with longer sessions are 2x more likely to buy.”

8. **Deployment**:
   - Output propensity scores daily for users.
   - Integrate into PlayStore for personalized recommendations (e.g., target high-propensity users with promotions).
   - Monitor model drift (e.g., changes in purchase behavior).

**Example**:
- Dataset: 1M users, 5% purchased.
- Features: `past_spend`, `session_time`, `app_category`.
- Model: XGBoost with class weights, AUC-ROC = 0.85.
- Action: Target top 10% of users, increasing conversions by 20%.

**Interview Tips**:
- Emphasize **business impact**: “The model helps prioritize marketing spend.”
- Discuss **imbalanced data**: “Purchases are rare, so I’d use class weights or SMOTE.”
- Be ready to sketch feature engineering or explain model choice trade-offs.
- Mention scalability: “For millions of users, I’d optimize feature computation.”

---

## 2. How would you build a fraud model without labels?

**Question**: [PayPal] Describe how you would build a fraud detection model when no labeled data (fraud vs. non-fraud) is available.

**Answer**:

Building a fraud detection model without labels requires **unsupervised** or **semi-supervised** learning, as we can’t rely on explicit fraud/non-fraud labels. Fraud is typically rare and anomalous, so we’ll treat it as an outlier detection problem. Here’s the approach:

1. **Problem Definition**:
   - **Objective**: Identify transactions that deviate from normal behavior, likely indicating fraud.
   - **Output**: Anomaly scores or binary flags (anomalous vs. normal).
   - **Metric**: Since no labels, use proxy metrics like anomaly rate or manual review feedback.

2. **Data Exploration**:
   - **Dataset**: Assume columns like `user_id`, `transaction_amount`, `timestamp`, `merchant`, `location`, `device_id`.
   - **EDA**: Analyze distributions (e.g., amount, frequency), detect patterns (e.g., typical user behavior), and check for missing values.

3. **Feature Engineering**:
   - **Transaction Features**:
     - Amount, time since last transaction, transaction frequency.
     - Merchant category, cross-border flag.
   - **User Behavior**:
     - Average spend, usual transaction locations.
     - Device changes, login frequency.
   - **Temporal Features**:
     - Hour of day, day of week (fraud may peak at odd hours).
   - **Aggregates**:
     - Rolling averages (e.g., spend over last 7 days).
     - Deviations (e.g., current amount vs. user’s average).

4. **Data Preprocessing**:
   - **Scaling**: Standardize numerical features (e.g., amount, frequency) for distance-based algorithms.
   - **Encoding**: One-hot encode categorical features (e.g., merchant type).
   - **Missing Values**: Impute or flag missing data (e.g., missing location as a feature).

5. **Model Selection**:
   - **Unsupervised**:
     - **Isolation Forest**: Efficient for high-dimensional data, assumes anomalies are “isolated.”
     - **DBSCAN**: Clusters normal transactions, flags outliers as noise.
     - **Autoencoders**: Learn to reconstruct normal transactions; high reconstruction error indicates anomalies.
   - **Semi-Supervised** (if partial labels emerge):
     - Train on “normal” data (assume most transactions are non-fraudulent).
     - Flag deviations as potential fraud.
   - **Why Isolation Forest**: Fast, scalable, and effective for rare anomalies like fraud.

6. **Training and Evaluation**:
   - **Training**: Fit the model on the full dataset (unsupervised) or assumed normal data (semi-supervised).
   - **Hyperparameters**: Tune `contamination` (expected anomaly rate, e.g., 0.01%) for Isolation Forest.
   - **Evaluation**:
     - No labels, so inspect top anomalies manually or with domain experts.
     - Proxy metrics: Compare anomaly patterns (e.g., high amounts, unusual locations) to known fraud heuristics.
     - If labels become available (e.g., manual reviews), compute precision/recall.

7. **Model Interpretation**:
   - Highlight features driving anomalies (e.g., “Transaction of $5000 vs. user’s avg. $50”).
   - Provide explainable outputs for fraud investigators (e.g., anomaly score + key features).

8. **Deployment**:
   - Score transactions in real-time for flagging.
   - Implement a feedback loop: Manual reviews refine the model (e.g., confirmed fraud becomes labels).
   - Monitor for drift (e.g., new fraud patterns).

**Example**:
- Dataset: 10M transactions, no labels.
- Model: Isolation Forest flags 0.1% as anomalies.
- Findings: Anomalies include large cross-border transactions at 3 AM.
- Action: Flag for review, reducing fraud risk.

**Interview Tips**:
- Emphasize **unsupervised learning**: “Without labels, I’d use anomaly detection.”
- Discuss **feedback loops**: “Manual reviews can create labels for future supervised models.”
- Mention **scalability**: “Isolation Forest handles millions of transactions efficiently.”
- Be ready to pivot: “If partial labels emerge, I’d switch to semi-supervised.”

---

## 3. How would you identify meaningful segmentation?

**Question**: [Apple] Explain how you would identify meaningful customer segments in a dataset.

**Answer**:

**Customer segmentation** divides users into groups with similar characteristics for targeted strategies (e.g., marketing, product design). “Meaningful” segments are actionable, distinct, and aligned with business goals. Here’s a step-by-step approach:

1. **Problem Definition**:
   - **Objective**: Group customers into clusters based on behavior, demographics, or preferences.
   - **Output**: Cluster labels for each customer, with interpretable segment profiles.
   - **Metric**: Silhouette score (cluster cohesion), business KPIs (e.g., revenue per segment).

2. **Data Exploration**:
   - **Dataset**: Assume columns like `user_id`, `age`, `location`, `purchase_history`, `app_usage`, `preferences`.
   - **EDA**: Check distributions (e.g., age, spend), correlations, and missing values. Identify key variables for segmentation.

3. **Feature Engineering**:
   - **Demographics**: Age, income, location (urban/rural).
   - **Behavioral**:
     - Frequency/recency of purchases, avg. spend.
     - App engagement (e.g., sessions per week, features used).
   - **Preferences**: Favorite categories, survey responses.
   - **Aggregates**: RFM scores (recency, frequency, monetary).
   - Standardize features to ensure equal weighting.

4. **Data Preprocessing**:
   - **Scaling**: Normalize numerical features (e.g., spend, sessions) for distance-based clustering.
   - **Encoding**: One-hot encode categorical features (e.g., location).
   - **Dimensionality Reduction**: Use PCA if high-dimensional to reduce noise while preserving variance.

5. **Model Selection**:
   - **Algorithms**:
     - **K-Means**: Simple, effective for spherical clusters.
     - **Hierarchical Clustering**: Captures nested structures, interpretable dendrograms.
     - **Gaussian Mixture Models (GMM)**: Flexible for non-spherical clusters.
     - **DBSCAN**: Identifies outliers, but may not suit all segmentation tasks.
   - **Why K-Means**: Fast, scalable, and works well with clear business features (e.g., spend, engagement).
   - **Number of Clusters**: Use elbow method (within-cluster variance) or silhouette score to choose `k`.

6. **Training and Evaluation**:
   - **Training**: Run clustering (e.g., K-Means with `k=4`) on preprocessed data.
   - **Evaluation**:
     - **Quantitative**: Silhouette score (higher = better separation), within-cluster variance.
     - **Qualitative**: Inspect segment profiles (e.g., “high-spend, young users”).
     - **Business**: Validate with KPIs (e.g., segment A drives 50% revenue).
   - Iterate on `k` or features if segments aren’t actionable.

7. **Interpretation**:
   - Profile segments: “Segment 1: Young, frequent buyers; Segment 2: Older, low engagement.”
   - Visualize: Scatter plots (PCA components) or bar charts (avg. spend per segment).
   - Map to business actions: “Target Segment 1 with premium offers.”

8. **Deployment**:
   - Assign new customers to segments using trained model.
   - Integrate into CRM for personalized campaigns.
   - Monitor segment stability over time (e.g., re-cluster quarterly).

**Example**:
- Dataset: 100K Apple users.
- Features: `age`, `spend`, `app_usage`.
- Model: K-Means, `k=3`.
- Segments: “Power users” (high spend, frequent), “Casual users” (moderate), “Inactive” (low).
- Action: Upsell to power users, re-engage inactives.

**Interview Tips**:
- Emphasize **business alignment**: “Segments must drive actionable strategies.”
- Discuss **interpretability**: “I’d ensure clusters are explainable to stakeholders.”
- Mention **validation**: “Silhouette score ensures clusters are distinct.”
- Be ready to pivot: “If segments aren’t meaningful, I’d adjust features or k.”

---

## Notes

- **Practicality**: Answers outline actionable steps, bridging ML and business needs.
- **Clarity**: Explanations are structured for verbal delivery, with clear methodology and trade-offs.
- **Depth**: Includes technical details (e.g., SMOTE, Isolation Forest) and business context (e.g., revenue impact).
- **Consistency**: Matches the style of `ml-coding.md`, `ml-theory.md`, and `ml-algorithms.md` for a cohesive repository.

For deeper practice, try coding a propensity model (see [ML Coding](ml-coding.md)) or explore [ML System Design](ml-system-design.md) for scaling these solutions. 🚀

---

**Next Steps**: Build on these skills with [ML System Design](ml-system-design.md) for production-ready solutions or revisit [ML Theory](ml-theory.md) for foundational concepts! 🌟