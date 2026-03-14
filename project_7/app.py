from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
import glob

app = FastAPI()

best_model = None
metrics = {}

class UserBehavior(BaseModel):
    calls: float
    minutes: float
    messages: float
    mb_used: float

def get_latest_model_paths():
    # Construct an absolute path to the models directory
    # so we can find any model regardless of where app.py is run from
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(models_dir):
        return None, None
        
    # Find ANY .joblib model in the models/ folder
    jobfiles = glob.glob(os.path.join(models_dir, '*.joblib'))
    if not jobfiles:
        return None, None
        
    # Get the most recently modified model, ensuring flexibility if the name changes
    latest_model = max(jobfiles, key=os.path.getmtime)
    
    model_name = os.path.basename(latest_model).replace('.joblib', '')
    metadata_path = os.path.join(models_dir, f"{model_name}_metadata.json")
    
    return latest_model, metadata_path

@app.on_event("startup")
def load_model():
    global best_model, metrics
    
    model_path, metadata_path = get_latest_model_paths()
    if not model_path:
        print("No model file found in models/. Please run main.py first.")
        return

    try:
        best_model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metrics.update(json.load(f))
            print(f"Metadata loaded successfully from {metadata_path}")
        except Exception as e:
            print(f"Error loading metadata: {e}")
    else:
        print(f"Metadata file {metadata_path} not found.")

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
        prediction = best_model.predict(input_data)[0]
        result = {"is_ultra_prediction": int(prediction)}
        # Log to file
        os.makedirs("docs", exist_ok=True)
        log_entry = {"input": behavior.model_dump(), "prediction": result["is_ultra_prediction"]}
        with open("docs/api_logs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
