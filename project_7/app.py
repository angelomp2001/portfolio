from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os

app = FastAPI()

# Load model and metadata at startup
MODEL_PATH = 'models/best_model.joblib'
METADATA_PATH = 'models/metadata.json'

if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
    best_model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    best_accuracy_score = metadata['best_accuracy_score']
    average = metadata['average']
    model_performance = metadata['model_performance']
    baseline_accuracy = metadata['baseline_accuracy']
else:
    # Fallback or error if model hasn't been trained yet
    print("Warning: Model or metadata not found. Please run 'python main.py' first.")
    best_model = None
    best_accuracy_score = 0
    average = 0
    model_performance = 0
    baseline_accuracy = 0

class UserBehavior(BaseModel):
    calls: float
    minutes: float
    messages: float
    mb_used: float

@app.get("/")
def get_model_performance():
    if best_model is None:
        return {"error": "Model not trained. Run main.py first."}
    return {
        "best_accuracy_score": float(best_accuracy_score),
        "average_target": float(average),
        "model_performance_ratio": float(model_performance),
        "baseline_accuracy": float(baseline_accuracy)
    }

@app.post("/predict")
def predict(behavior: UserBehavior):
    if best_model is None:
        return {"error": "Model not trained. Run main.py first."}
    # Convert input to DataFrame
    input_data = pd.DataFrame([behavior.model_dump()])
    prediction = best_model.predict(input_data)
    return {"is_ultra_prediction": int(prediction[0])}
