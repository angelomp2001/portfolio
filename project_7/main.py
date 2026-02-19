from src.data.loader import load_data
from src.data.explorer import DataExplorer
from src.pipeline import select_best_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import joblib
import json
import os

def main():
    #import data
    path = 'data/users_behavior.csv'
    df = load_data(path)

    print(df.head())
    print(df.describe())

    # QC data quality
    explorer = DataExplorer(df)
    explorer.view() 

    # define target and features
    target = df['is_ultra']
    features = df.drop(target.name, axis = 1)

    # select best model
    # Replaces model_picker functionality
    best_model, best_accuracy_score, train_features, train_target, test_features, test_target = select_best_model(features, target)

    # sanity check using average
    average = train_target.mean()

    # model performance vs average
    model_performance = best_accuracy_score / average
    print(f'average:{average}\nmodel_performance:{model_performance}')

    # saninty check with DummyClassifier (creates column of target based on strategy and no features)
    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy_clf.fit(train_features, train_target) # it asks for features, but it doesn't use them. 
    dummy_y_hat = dummy_clf.predict(test_features)
    baseline_accuracy = accuracy_score(test_target, dummy_y_hat)

    print("Baseline Accuracy:", baseline_accuracy)

    # Log results to RESULTS.md
    with open("RESULTS.md", "a") as f:
        f.write(f"| Experiment | Accuracy | Baseline | Performance Ratio |\n")
        f.write(f"| Refactor-Data-Preprocessing | {best_accuracy_score:.4f} | {baseline_accuracy:.4f} | {model_performance:.4f} |\n")
        
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Save the model
    MODEL_PATH = 'models/best_model.joblib'
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save the metadata
    metadata = {
        "best_accuracy_score": float(best_accuracy_score),
        "average": float(average),
        "model_performance": float(model_performance),
        "baseline_accuracy": float(baseline_accuracy)
    }
    METADATA_PATH = 'models/metadata.json'
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    main()
