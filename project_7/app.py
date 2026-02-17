from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# Global variables for model and metrics
best_model = None
metrics = {}

class UserBehavior(BaseModel):
    calls: float
    minutes: float
    messages: float
    mb_used: float

@app.on_event("startup")
def load_model():
    global best_model, metrics
    model_path = "model.joblib"
    if os.path.exists(model_path):
        best_model = joblib.load(model_path)
        print("Model loaded successfully.")
        
        # In a real scenario, you might save metrics to a json file alongside the model
        # For now, we'll hardcode or read from a file if available.
        # Since the user wants speed, reloading metrics from main execution is risky if it triggers training.
        # Let's assume metrics are static for this deployment or read from a simple sidecar file if created.
        # To keep it simple and fast, we will allow the / endpoint to return unavailable or read from a log.
        # But to match previous behavior, I'll hardcode the latest seen values or we can save them to a json.
        
        # Let's try to load metrics from a file strictly for metrics if we want to be clean.
        # Or just return a message saying "check logs". 
        # But the user asked why it's slow. It was slow because it was TRAINING on import.
        # Now it just loads.
    else:
        print(f"Model file {model_path} not found. Please run main.py first.")

@app.get("/")
def get_model_performance():
    # Since we decoupled training, these variables aren't directly available unless saved.
    # For now, we will return a status message.
    if best_model:
        return {"status": "Model loaded", "message": "Metrics are not persisted in this lightweight app version. Check RESULTS.md"}
    return {"status": "Model not loaded", "message": "Please run main.py to train and save the model."}

@app.post("/predict")
def predict(behavior: UserBehavior):
    if not best_model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    # Convert input to DataFrame
    input_data = pd.DataFrame([behavior.model_dump()])
    prediction = best_model.predict(input_data)
    return {"is_ultra_prediction": int(prediction[0])}
