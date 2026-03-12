# src/pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.loader import load_data
from src.data.explorer import DataExplorer
from src.models.trainer import ModelTrainer, split_data
from src.models.tuner import HyperparameterTuner
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.config import RANDOM_STATE, NUMERIC_COLS
import os

def select_best_model(features, target):
    """
    Select the best model from a set of candidates, tuned on validation data.
    """
    df = pd.concat([features, target], axis=1)
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(df, target.name, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_valid[NUMERIC_COLS] = scaler.transform(X_valid[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])
    
    models_config = [
        {
            'name': 'DecisionTreeClassifier',
            'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'param_grid': {'max_depth': list(range(1, 21))}
        },
        {
            'name': 'RandomForestClassifier',
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'param_grid': {'max_depth': list(range(1, 21)), 'n_estimators': [10, 50, 100]}
        },
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(random_state=RANDOM_STATE, solver='liblinear', max_iter=200),
            'param_grid': {} 
        }
    ]
    
    best_overall_model = None
    best_overall_score = -1
    
    for config in models_config:
        print(f"Processing {config['name']}...")
        if config['param_grid']:
            # model trainer with hyperparameter tuning via RandomizedSearchCV
            tuner = HyperparameterTuner(config['model'], config['param_grid'], cv=3)
            best_model_training, score = tuner.tune(X_train, y_train)
        else:
            # model trainer without hyperparameter tuning
            trainer = ModelTrainer(config['model'], random_state=RANDOM_STATE)
            trainer.train(X_train, y_train)
            best_model_training = config['model']
        
        val_score = best_model_training.score(X_valid, y_valid)
        print(f"  Validation Score: {val_score:.4f}")
        
        if val_score > best_overall_score:
            best_overall_score = val_score
            best_overall_model = best_model_training
            
    # Compute extra metrics on best model
    preds = best_overall_model.predict(X_test)
    preds_proba = best_overall_model.predict_proba(X_test)[:, 1] if hasattr(best_overall_model, "predict_proba") else preds
    
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, preds_proba)
    }
    print("Final Test Metrics:", metrics)
    
    # Random Forest / Decision Tree specific exports
    if isinstance(best_overall_model, RandomForestClassifier):
        with open("docs/random_forest_important_features.md", "w") as f:
             f.write(pd.Series(best_overall_model.feature_importances_, index=features.columns).sort_values(ascending=False).to_markdown())
    if isinstance(best_overall_model, DecisionTreeClassifier):
        plt.figure(figsize=(20,10))
        plot_tree(best_overall_model, feature_names=features.columns, filled=True)
        plt.savefig("docs/decision_tree_model.png")
        plt.close()
    
    return best_overall_model, best_overall_score, X_train, y_train, X_test, y_test, metrics

def run_pipeline(file_path, target_col):
    print("Loading data...")
    df = load_data(file_path)
    
    print("Exploring data...")
    explorer = DataExplorer(df)
    explorer.view()
    
    target = df[target_col]
    features = df.drop(target_col, axis=1)
    
    best_model, score, _, _, X_test, y_test, metrics = select_best_model(features, target)
    print(f"\nBest Overall Model: {best_model}")
    return best_model, score, metrics

if __name__ == "__main__":
    path = 'data/users_behavior.csv'
    if not os.path.exists(path):
        potential_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'users_behavior.csv')
        if os.path.exists(potential_path):
             path = potential_path
    try:
        run_pipeline(path, 'is_ultra')
    except Exception as e:
        print(f"An error occurred: {e}")
