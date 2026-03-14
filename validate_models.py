import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sys
import os

# Add both projects to path just in case we need their modules
sys.path.append(os.path.abspath('project_7'))
sys.path.append(os.path.abspath('project_7_v2'))

def validate_models():
    # Load dataset
    df = pd.read_csv('project_7/data/data.csv')
    
    # 1. Prepare data for project 7
    # project_7 uses train/valid/test split and fits StandardScaler on train set. 
    # To truly replicate project_7's exact scaling, we have to replicate its split exactly.
    from project_7.src.models.trainer import split_data
    X_train_7, _, _, _, X_test_7, y_test_7 = split_data(df, 'is_ultra', random_state=12345)
    
    scaler = StandardScaler()
    num_cols = ['calls', 'minutes', 'messages', 'mb_used']
    scaler.fit(X_train_7[num_cols])
    
    # Let's just predict on the entire dataset for a thorough comparison
    X_all_7 = df.drop('is_ultra', axis=1).copy()
    X_all_7[num_cols] = scaler.transform(X_all_7[num_cols])
    
    model_7 = joblib.load('project_7/models/best_model.joblib')
    # wait, best_model_training is just the classifier. Let's predict.
    preds_7_all = model_7.predict(X_all_7)
    
    # 2. Prepare data for project_7_v2
    # project_7_v2 pipeline includes preprocessing.
    X_all_v2 = df.drop('is_ultra', axis=1).copy()
    
    model_v2 = joblib.load('project_7_v2/models/RandomForestClassifier.joblib')
    preds_v2_all = model_v2.predict(X_all_v2)
    
    # 3. Compare Results
    match_rate = np.mean(preds_7_all == preds_v2_all)
    
    print("--- Model Validation ---")
    print(f"Project 7 model type: {type(model_7)}")
    print(f"Project 7 v2 model type: {type(model_v2)}")
    print(f"Predictions Match Rate on full dataset: {match_rate:.2%}")
    print(f"Total matching predictions: {np.sum(preds_7_all == preds_v2_all)} / {len(df)}")
    
if __name__ == '__main__':
    validate_models()
