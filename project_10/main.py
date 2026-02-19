from src.data_preprocessing import (
    load_data,
    preprocess_data,
    verify_recovery_calculation,
    analyze_metal_concentrations,
    compare_feed_distributions,
    remove_anomalies,
    train_and_evaluate_models,
    train_and_save_best_model
)
from src.data_explorers import view

# ==================================================================================================
# 1. LOAD DATA
# ==================================================================================================
print("1. Loading Data...")
gold_recovery_full = load_data('data/gold_recovery_full.csv')
gold_recovery_train = load_data('data/gold_recovery_train.csv')
gold_recovery_test = load_data('data/gold_recovery_test.csv')


# ==================================================================================================
# 2. PREPROCESS DATA (Basic Cleaning)
# ==================================================================================================
# Raw data often contains missing values or incorrect data types (e.g., dates as strings).
# We need to convert dates to datetime objects and drop rows with missing values (dropna) to ensure
# models receive clean input. Note: Dropping data is acceptable here due to dataset size,
# but imputation could be considered in future iterations.
print("\n2. Preprocessing Data...")
gold_recovery_full, gold_recovery_train, gold_recovery_test = preprocess_data(
    gold_recovery_full, gold_recovery_train, gold_recovery_test
)


# ==================================================================================================
# 3. VERIFY RECOVERY CALCULATION
# ==================================================================================================
# The target variable `rougher.output.recovery` may contain bad data. We will manually calculate 
# recovery using the chemical formula and input features, then compare it to the provided value 
# using Mean Absolute Error (MAE). There should be no difference.
print("\n3. Verifying Recovery Calculation...")
# Note: show_plot=True will pop up the window. 
# For API, we will use the returned figure. Here we just let it show if run interactively.
verify_recovery_calculation(gold_recovery_train, show_plot=False) 


# ==================================================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA): Metal Concentrations
# ==================================================================================================
# We need to trust the data reflects physical reality. Does gold concentration actually
# increase as the ore goes through the process? We will plot the distribution of 
# Au (Gold), Ag (Silver), and Pb (Lead) at each purification stage.
# We expect Au to increase and impurities (Ag, Pb) to decrease or stabilize.
print("\n4. Analyzing Metal Concentrations...")
analyze_metal_concentrations(gold_recovery_full, show_plot=False)


# ==================================================================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA): Feed Particle Size
# ==================================================================================================
# Machine learning models assume the Train and Test data come from the same distribution.
# If the input feed size is different in Test vs Train, the model's predictions may fail.
# We will compare the feed size distributions of Train vs Test sets.
# If they are significantly different, the model evaluation might be unreliable.
print("\n5. Comparing Feed Particle Size Distributions...")
compare_feed_distributions(gold_recovery_train, gold_recovery_test)


# ==================================================================================================
# 6. FEATURE ENGINEERING & ANOMALY REMOVAL
# ==================================================================================================
# Problem: 1. Some rows have a total metal concentration of 0 (impossible in reality), indicating
#             bad sensor data.
#          2. We must define which columns are valid inputs for the model (columns present in both
#             Train and Test sets) to avoid "data leakage" (using future data).
# Solution: Calculate total concentrations and remove rows with 0 sum. Define `common_columns`
#           to ensure the model only sees features available at prediction time.
print("\n6. Removing Anomalies & Preparing Features...")
gold_recovery_train, gold_recovery_test, common_columns = remove_anomalies(
    gold_recovery_train, gold_recovery_test, gold_recovery_full
)


# ==================================================================================================
# 7. MODEL TRAINING & EVALUATION
# ==================================================================================================
# Problem: We need to predict two targets: `rougher.output.recovery` and `final.output.recovery`.
# Solution: Train three different regression models (Linear, Decision Tree, Random Forest) using
#           Cross-Validation.
#           Metric: sMAPE (Symmetric Mean Absolute Percentage Error).
#           Final Score: Weighted average (25% rougher, 75% final).
#           The function prints the sMAPE for each model/target and the final weighted score.
print("\n7. Training and Evaluating Models...")
train_and_evaluate_models(gold_recovery_train, common_columns)

# ==================================================================================================
# 8. SAVE BEST MODEL FOR API
# ==================================================================================================
# Save the trained Random Forest model (found to be best in step 7) so the API can use it
# without retraining.
print("\n8. Saving Best Model for API...")
train_and_save_best_model(gold_recovery_train, common_columns)

print("\nAnalysis Complete.")

