'''
This script demonstrates specific functionality of the project, such as KNN similarity search and data obfuscation.
These are auxiliary demos and not part of the main training pipeline.
'''

from src.data_preprocessing import *
import pandas as pd
from IPython.display import display

def run_demos(path='data/insurance_us.csv'):
    # load data
    df, features = load_and_label_label(path)
    
    ## scale features
    df_scaled = scale_features(df, features)

    print("\n--- Demo: Finding similar records (KNN) ---")
    ## finding similar records to a random sample of 5 customers
    # Ensure we don't sample more than available if dataset is small, though clear here.
    sample_indices = df_scaled.sample(min(5, len(df_scaled))).index
    
    for idx in sample_indices:
        print(f"\nNeighbors for customer index {idx}:")
        print("Unscaled - Euclidean")
        display(get_knn(df, features, row=idx, k=5, metric=2))

        print("Unscaled - Manhattan")
        display(get_knn(df, features, row=idx, k=5, metric=1))

        print("Scaled - Euclidean")
        display(get_knn(df_scaled, features, row=idx, k=5, metric=2))

        print("Scaled - Manhattan")
        display(get_knn(df_scaled, features, row=idx, k=5, metric=1))

    print("\n--- Demo: Data Obfuscation ---")
    ## obfuscating data
    personal_info_column_list = ['gender', 'age', 'income', 'family_members']
    df_pn = df[personal_info_column_list]

    X, X_transformed, P = obfuscate_data(df_pn, obfuscate = True)

    # recover obfuscated data
    X_recovered = obfuscate_data(X_transformed, obfuscate = False, P = P)

    # 3 tests to measure impact of obfuscation data on regression
    print("\nComparing Original, Transformed, and Recovered data for first 3 customers:")
    for i in range(3):
        print(f"\nCustomer {i+1}:")
        print("Original:   ", X[i])
        print("Transformed:", X_transformed[i])
        print("Recovered:  ", X_recovered[i])
        
    print("\n Demos completed.")

if __name__ == "__main__":
    run_demos()
