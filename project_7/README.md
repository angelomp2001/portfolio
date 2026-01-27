# User Behavior Model Analysis

## Project Overview
This project predicts whether a user is using the "Ultra" plan (`is_ultra`) based on their behavior data. We aim to develop a model with the highest possible accuracy, targeting a threshold of 0.75.

## Data Preprocessing & Modeling Strategy
The target variable (`is_ultra`) is imbalanced: roughly 30% are "Ultra" users (1s) and 70% are standard users (0s). We addressed this in two key ways:
1.  **Stratified Splitting**: We used the `stratify` parameter in `train_test_split()` to ensure our training, validation, and test sets all retain the original 30/70 proportion of target values.
2.  **Baseline Comparison**: We evaluated model quality not just by raw accuracy, but by comparing it to a "Dummy" baseline.

## Conclusion
Our analysis shows that accounting for the target distribution is critical.

*   **Baseline Model**: A naive "Dummy" predictor (which simply guesses the most frequent class) achieves an accuracy of ~**69%**. This represents the accuracy one gets by having "no model".
*   **Our Model**: After hyperparameter optimization, our best model (RandomForestClassifier) achieves a test accuracy of ~**81%**.

**Result**: Our model outperforms the baseline by **12%** (81% vs 69%). Given a success threshold of 5% outperformance, we exceeded this by over 2x, confirming the model provides significant predictive value.
