import os
import json
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Car Price Prediction API",
    description="API to predict car prices using a trained machine learning model.",
    version="1.0.0"
)

MODEL_PATH = "models/best_model.joblib"
LOG_PATH = "data/api_inference_logs.jsonl"

# Load the model globally at startup
try:
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    else:
        model_pipeline = None
        print(f"Warning: Model file not found at {MODEL_PATH}")
except Exception as e:
    model_pipeline = None
    print(f"Error loading model: {e}")

class CarFeatures(BaseModel):
    VehicleType: str
    RegistrationYear: float
    Gearbox: str
    Power: float
    Model: str
    Mileage: int
    FuelType: str
    Brand: str
    NotRepaired: str

@app.post("/predict")
def predict_price(features: CarFeatures):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded or unavailable.")
        
    try:
        # 1. Convert input to dict and DataFrame
        input_dict = features.model_dump()
        df = pd.DataFrame([input_dict])
        
        # 2. Make prediction
        prediction = model_pipeline.predict(df)[0]
        
        # 3. Log run and result
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_dict,
            "prediction": float(prediction)
        }
        
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        # 4. Return result
        return {"prediction": float(prediction)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_pipeline is not None}
