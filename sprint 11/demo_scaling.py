'''
Objective 3: Predict Number of Benefits (Linear Regression)

Demonstrates that feature scaling does NOT affect Linear Regression predictions
(RMSE/R2), because the model weights adjust inversely to the scale.
'''

from src.data_preprocessing import data_preprocessor, MyLinearRegression, eval_regressor


def run_scaling_demo(df, features, split_random_state):
    '''
    Fits a custom Linear Regression model on both unscaled and scaled features,
    then prints a side-by-side comparison of RMSE and R2.

    Returns the regression train/test splits and unscaled RMSE so that
    the obfuscation demo (Objective 4) can reuse them without re-splitting.

    :param df: Full DataFrame (must include 'insurance_benefits' column)
    :param features: List of feature column names
    :param split_random_state: Random state used for the train/test split
    :return: (X_train_reg, X_test_reg, y_train_reg, y_test_reg, rmse_unscaled)
    '''

    # Split using the CONTINUOUS target (number of benefits, not binary).
    # Same random_state as Objective 2 so Train/Test rows are identical.
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_train_scaled_reg, X_test_scaled_reg = data_preprocessor(
        df, features, target='insurance_benefits', test_size=0.3, random_state=split_random_state
    )

    lr = MyLinearRegression()

    # --- Unscaled ---
    lr.fit(X_train_reg, y_train_reg)
    y_pred_unscaled = lr.predict(X_test_reg)
    rmse_unscaled, r2_unscaled = eval_regressor(y_test_reg, y_pred_unscaled, print_metrics=False)

    # --- Scaled ---
    lr.fit(X_train_scaled_reg, y_train_reg)
    y_pred_scaled = lr.predict(X_test_scaled_reg)
    rmse_scaled, r2_scaled = eval_regressor(y_test_reg, y_pred_scaled, print_metrics=False)

    # --- Comparison Table ---
    print(f"{'Metric':<10} {'Unscaled':<10} {'Scaled':<10}")
    print("-" * 32)
    print(f"{'RMSE':<10} {rmse_unscaled:<10.2f} {rmse_scaled:<10.2f}")
    print(f"{'R2':<10} {r2_unscaled:<10.2f} {r2_scaled:<10.2f}")
    print("-" * 32)
    print("Conclusion: Weights change, but predictions (RMSE/R2) remain identical.")
    print("Note on Performance: The R2 score is low (~0.43), indicating Linear Regression is")
    print("not the best model for this data. However, the objective was to prove scaling invariance.")

    # Return splits + unscaled RMSE/R2 so Objective 4 can reuse without re-splitting
    return X_train_reg, X_test_reg, y_train_reg, y_test_reg, rmse_unscaled, r2_unscaled
