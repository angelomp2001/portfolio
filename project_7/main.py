from src.data_preprocessing import *
import joblib
import json
import os
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

def run_training():
    # import data
    path = 'data/users_behavior.csv'
    df = load_data(path)
    
    print("Initial Data Check:")
    print(df.head())
    print(df.describe())

    # QC data quality
    view(df, 'missing values')

    # define target and features
    target = df['is_ultra']
    features = df.drop(target.name, axis=1)

    # select best model
    print("\nStarting model selection and hyperparameter optimization...")
    best_model, best_accuracy_score, train_features, train_target, test_features, test_target = model_picker(features, target)

    # sanity check using average
    average = float(train_target.mean())

    # model performance vs average
    model_performance = float(best_accuracy_score / average)


# saninty check with DummyClassifier (creates column of target based on strategy and no features)
    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy_clf.fit(train_features, train_target)
    dummy_y_hat = dummy_clf.predict(test_features)
    baseline_accuracy = float(accuracy_score(test_target, dummy_y_hat))

    print(f"\nTraining Complete.")
    print(f"Average Target Rate: {average}")
    print(f"Model Performance Factor: {model_performance}")
    print(f"Baseline (Dummy) Accuracy: {baseline_accuracy}")

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Save the model
    joblib.dump(best_model, 'models/best_model.joblib')
    
    # Save the metadata
    metadata = {
        "best_accuracy_score": float(best_accuracy_score),
        "average": average,
        "model_performance": model_performance,
        "baseline_accuracy": baseline_accuracy
    }
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\nModel and metadata saved to 'models/' directory.")

if __name__ == "__main__":
    run_training()
