import pandas as pd
from src.data.loader import load_data
from src.data.explorer import DataExplorer
from src.models.trainer import ModelTrainer, split_data
from src.models.tuner import HyperparameterTuner
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def select_best_model(features, target):
    """
    Select the best model from a set of candidates, tuned on validation data.
    """
    # 1. Split Data
    # Concatenate to ensure alignment during split if features/target come from same df
    df = pd.concat([features, target], axis=1)
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(df, target.name)
    
    # 2. Define Models and Grids
    models_config = [
        {
            'name': 'DecisionTreeClassifier',
            'model': DecisionTreeClassifier(random_state=12345),
            'param_grid': {'max_depth': list(range(1, 21))}
        },
        {
            'name': 'RandomForestClassifier',
            'model': RandomForestClassifier(random_state=12345),
            'param_grid': {'max_depth': list(range(1, 21)), 'n_estimators': [10, 50, 100]}
        },
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(random_state=12345, solver='liblinear', max_iter=200),
            'param_grid': {} 
        }
    ]
    
    best_overall_model = None
    best_overall_score = -1
    
    for config in models_config:
        print(f"Processing {config['name']}...")
        if config['param_grid']:
            tuner = HyperparameterTuner(config['model'], config['param_grid'], cv=3)
            # Use training set for tuning
            best_model, score = tuner.tune(X_train, y_train)
        else:
            trainer = ModelTrainer(config['model'])
            trainer.train(X_train, y_train)
            best_model = config['model']
        
        # Validate on hold-out validation set
        val_score = best_model.score(X_valid, y_valid)
        print(f"  Validation Score: {val_score:.4f}")
        
        if val_score > best_overall_score:
            best_overall_score = val_score
            best_overall_model = best_model
            
    # Refit isn't strictly necessary if strict separation is maintained, 
    # but to match original logic which trained on X_train and evaluated on X_test:
    test_score = best_overall_model.score(X_test, y_test)
    print(f"Final Test Score: {test_score:.4f}")
    
    return best_overall_model, test_score, X_train, y_train, X_test, y_test

def run_pipeline(file_path, target_col):
    # 1. Load Data
    print("Loading data...")
    df = load_data(file_path)
    
    # 2. Explore Data
    print("Exploring data...")
    explorer = DataExplorer(df)
    explorer.view()
    
    target = df[target_col]
    features = df.drop(target_col, axis=1)
    
    best_model, score, _, _, X_test, y_test = select_best_model(features, target)
    
    print(f"\nBest Overall Model: {best_model}")
    print(f"Validation Score: {score}")
    
    # 5. Final Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    test_score = best_model.score(X_test, y_test)
    print(f"Test Score: {test_score}")
    
    return best_model, test_score

if __name__ == "__main__":
    # Example usage
    path = 'data/users_behavior.csv' # Adjust path as needed
    # Check if file exists relative to where script is run. 
    # Attempt to locate it.
    import os
    if not os.path.exists(path):
        # Try finding it in project root
        potential_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'users_behavior.csv')
        if os.path.exists(potential_path):
             path = potential_path
            
    try:
        run_pipeline(path, 'is_ultra')
    except Exception as e:
        print(f"An error occurred: {e}")
