from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os

# Create the app instance
app = FastAPI()

# Global variables for model and metrics
best_model = None
metrics = {
    "best_accuracy_score": 0.0,
    "average": 0.0,
    "model_performance": 0.0,
    "baseline_accuracy": 0.0
}

MODEL_PATH = 'models/best_model.joblib'
METADATA_PATH = 'models/metadata.json'

class UserBehavior(BaseModel):
    calls: float
    minutes: float
    messages: float
    mb_used: float

@app.on_event("startup")
def load_model():
    global best_model, metrics
    
    if os.path.exists(MODEL_PATH):
        try:
            best_model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file {MODEL_PATH} not found. Please run main.py first.")

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r') as f:
                metrics.update(json.load(f))
            print(f"Metadata loaded successfully from {METADATA_PATH}")
        except Exception as e:
            print(f"Error loading metadata: {e}")
    else:
        print(f"Metadata file {METADATA_PATH} not found.")

@app.get("/")
def get_model_performance():
    if not best_model:
         return {"status": "Model not loaded", "message": "Please run main.py to train and save the model."}
    
    return {
        "status": "Model loaded",
        "metrics": metrics
    }

@app.post("/predict")
def predict(behavior: UserBehavior):
    if not best_model:
        raise HTTPException(status_code=503, detail="Model not loaded. Please run main.py first.")
    
    # Convert input to DataFrame
    input_data = pd.DataFrame([behavior.model_dump()])
    try:
        prediction = best_model.predict(input_data)
        return {"is_ultra_prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    import uvicorn
    # Allow running the app directly with python app.py
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
