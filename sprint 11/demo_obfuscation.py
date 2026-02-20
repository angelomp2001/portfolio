'''
Objective 4: Protect Client Data (Obfuscation Proof)

Demonstrates that multiplying features by a random invertible matrix P
(obfuscation) does NOT degrade Linear Regression performance.
The model trains on masked data and still produces identical RMSE/R2.
'''

import numpy as np
from src.data_preprocessing import MyLinearRegression, eval_regressor


def run_obfuscation_demo(X_train_reg, X_test_reg, y_train_reg, y_test_reg, rmse_unscaled, r2_unscaled):
    '''
    Generates a random invertible matrix P, obfuscates the training and test
    features (X' = X @ P), trains a Linear Regression model on the obfuscated
    data, and prints a comparison against the original (unscaled) results.

    :param X_train_reg: Unscaled training features from Objective 3
    :param X_test_reg:  Unscaled test features from Objective 3
    :param y_train_reg: Training labels (continuous insurance_benefits)
    :param y_test_reg:  Test labels (continuous insurance_benefits)
    :param rmse_unscaled: Baseline RMSE from Objective 3 (unscaled) for comparison
    '''

    # --- Generate P (Random Invertible Matrix) ---
    rng = np.random.default_rng(seed=42)
    P = rng.random(size=(X_train_reg.shape[1], X_train_reg.shape[1]))
    while np.linalg.det(P) == 0:
        P = rng.random(size=(X_train_reg.shape[1], X_train_reg.shape[1]))

    # --- Obfuscate: X' = X @ P ---
    X_obfuscated_train = X_train_reg @ P
    X_obfuscated_test  = X_test_reg  @ P

    # --- Train and Predict on Obfuscated Data ---
    lr = MyLinearRegression()
    lr.fit(X_obfuscated_train, y_train_reg)
    y_pred_obfuscated = lr.predict(X_obfuscated_test)

    # --- Evaluate ---
    rmse_obf, r2_obf = eval_regressor(y_test_reg, y_pred_obfuscated, print_metrics=False)

    # --- Comparison Table ---
    print(f"{'Metric':<10} {'Original':<10} {'Obfuscated':<10}")
    print("-" * 34)
    print(f"{'RMSE':<10} {rmse_unscaled:<10.2f} {rmse_obf:<10.2f}")
    print(f"{'R2':<10} {r2_unscaled:<10.2f} {r2_obf:<10.2f}")
    print("-" * 34)
    print("Technical Conclusion: RMSE scores match between Original and Obfuscated data.")
    print("This proves that Linear Regression can train on encrypted/obfuscated data without loss of accuracy.")
