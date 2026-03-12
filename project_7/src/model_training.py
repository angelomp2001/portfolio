import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from src.config import TARGET_COL

def get_scorer(metric):
    """Return a scorer based on string name"""
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score)
    }
    return scorers.get(metric, make_scorer(accuracy_score))

def train_model(pipeline, X_train, y_train, k_folds, random_state, metric, param_grid):
    """
    Train and evaluate model using cross validation. Return CV results.
    """
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
    )
    
    cv_results = {}
    for k, v in scores.items():
        if k.startswith('test_'):
            cv_results[k.replace('test_', '')] = v.mean()
    
    return cv_results

def tune_model(pipeline, X_train, y_train, metric, param_grid):
    """
    Hyperparameter search on best model
    """
    scorer = get_scorer(metric)
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=10, 
        scoring=scorer,
        cv=3,
        random_state=42, # config.RANDOM_STATE
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def save_model(model, model_name, metadata):
    """
    Save best model pipeline and metadata
    """
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{model_name}.joblib')
    
    with open(f'models/{model_name}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

def test_model(model, X_test, y_test, metric):
    """
    Test evaluation of best tuned pipeline
    """
    y_pred = model.predict(X_test)
    
    # Calculate probabilities if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback if no probabilities (e.g. some SVMS)
        y_prob = model.decision_function(X_test) if hasattr(model, "decision_function") else y_pred
        
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    return results

def save_results(results, model_name):
    """
    Save the results mapping.
    """
    os.makedirs('docs', exist_ok=True)
    
    # Save the RESULTS.md
    with open('docs/RESULTS.md', 'w') as f:
        f.write("# Results\n\n")
        f.write(f"**Best Model**: {model_name}\n")
        f.write(f"**Primary Metric ({results['metric']}) Test Score**: {results['test'][results['metric']]:.4f}\n\n")
        
        f.write("### Test Set Evaluation\n")
        f.write("| Metric | Score |\n")
        f.write("|---|---|\n")
        for k, v in results['test'].items():
            f.write(f"| {k} | {v:.4f} |\n")
            
        f.write("\n### CV Results for Best Model\n")
        f.write("| Metric | Mean CV Score |\n")
        f.write("|---|---|\n")
        best_cv = results['cv'][results['best_model_name']]
        for k, v in best_cv.items():
            f.write(f"| {k} | {v:.4f} |\n")
