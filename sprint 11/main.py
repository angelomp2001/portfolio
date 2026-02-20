'''
Marketing needs to predict who's likely to be a custom, receive benefits, how many benefits, while protecting their data. 
'''

import argparse
from src.data_preprocessing import *
from demo_scaling import run_scaling_demo
from demo_obfuscation import run_obfuscation_demo

def main():
    parser = argparse.ArgumentParser(description="Insurance Solutions Analysis")
    parser.add_argument('--eda', action='store_true', help="Run Exploratory Data Analysis")
    args = parser.parse_args()

    # load data
    path = 'data/insurance_us.csv'
    try:
        df, features = load_and_label_label(path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return

    # EDA
    if args.eda:
        print("Running Exploratory Data Analysis...")
        EDA(df, features)
        print("EDA Plot generated (if in interactive environment) or saved.")
        # Note: generic pairplot might block execution if plt.show() is implied or backend dependent, 
        # but in this script structure it's just a function call.

    # Objective 1: Find Similar Customers. 
    # We will be identifying which rows (customers) in the dataframe are closest to a specific target row/customer.
    # We use the K-Nearest Neighbors (KNN) algorithm, which calculates the distance (Euclidean/Manhattan) between rows.

    # 1. Feature Scaling
    # In order to accurately compare distance between rows, all features must be on the same scale.
    # If one feature (Income) ranges 0-100,000 and another (Age) ranges 0-100
    # the sheer size of the "Income" difference will overwhelm the "Age" difference in the distance calculation.
    # So we scale all features to a standard range (0 to 1) so that every feature (Age, Income, etc.) 
    # contributes equally to the similarity score.
    df_scaled = scale_features(df, features)

    # 2. Target Label Processing (Probability of receiving benefits)
    # The original dataset contains the *number* of benefits received (0, 1, 2...)
    # not whether or not they received benefits. Objective 2 is to predict whether or not
    # they receive ANY benefit.
    # So we convert the continuous variable 'insurance_benefits' to a binary one (1 if > 0, else 0).
    target_s, target = continuous_to_binary(df['insurance_benefits'])
    df['insurance_benefits_binary'] = target_s # Save binary target to df
    
    # We use same random state for both classification and regression splits
    # to ensure they are comparable (same rows in train/test).
    split_random_state = 12345

    # Objective 1: Find Similar Customers.
    print("\n--- Objective 1: Find Similar Customers (KNN) ---")
    print("Finding the 5 nearest neighbors for a sample customer (Index 0):")
    
    # We use the scaled data because we established that scaling is crucial.
    # We find 5 neighbors for a sample customer in the dataset. 
    # The sample customer is the first customer in the dataset (Index 0).
    sample_idx = 0
    # metric=2 corresponds to Euclidean distance in NearestNeighbors(p=metric)
    neighbors_scaled = get_knn(df_scaled, features, row=sample_idx, k=5, metric=2)
    
    # We have the result, but we want to display the unscaled values for readability.
    # Since the indices are preserved, we fetch rows from original df using the results indices.
    neighbors_original = df.loc[neighbors_scaled.index].copy()
    
    # We add the distance column back (from the scaled calculation)
    neighbors_original['distance'] = neighbors_scaled['distance']
    
    # Display the results (Neighbors should look similar to the target)
    print(f"Target Customer (Index {sample_idx}):")
    print(df.iloc[sample_idx][features].to_string())
    print("\nNearest Neighbors (Original Values):")
    print(neighbors_original)

    # Objective 2: Predict Benefit Receipt (Classification - Yes/No)
    # We split the data into training and test sets.
    # IMPORTANT: We use the *binary* target column created above for classification.
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = data_preprocessor(
        df, features, target='insurance_benefits_binary', test_size=0.3, random_state=split_random_state
    )

    print("\n--- Objective 2: Predict Benefit Receipt (KNN Classification) ---")
    # We calculate and compare the F1 score of KNN on unscaled vs. scaled data.
    f1_knn(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, k=5)

    # Objective 3: Predict Number of Benefits (Linear Regression)
    print("\n--- Objective 3: Predict Number of Benefits (Linear Regression) ---")
    # we will predict the number of benefits using linear regression and also test the effect of scaling on the model.
    # See demo_scaling.py for full implementation details.
    # Returns regression splits for reuse in Objective 4, plus the baseline unscaled RMSE.
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, rmse_unscaled, r2_unscaled = run_scaling_demo(
        df, features, split_random_state
    )
    
    # Objective 4: Protect Client Data (Obfuscation Proof)
    print("\n--- Objective 4: Protect Client Data (Obfuscation Proof) ---")
    # we will demonstrate how obfuscation can be used to protect client data while preserving model accuracy.
    # See demo_obfuscation.py for full implementation details.
    # Reuses the regression splits and baseline RMSE/R2 from Objective 3.
    run_obfuscation_demo(X_train_reg, X_test_reg, y_train_reg, y_test_reg, rmse_unscaled, r2_unscaled)

if __name__ == "__main__":
    main()