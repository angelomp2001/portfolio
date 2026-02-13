from fastapi import FastAPI
from pydantic import BaseModel
from main import best_model, best_accuracy_score, average, model_performance, baseline_accuracy
import pandas as pd

app = FastAPI()

class UserBehavior(BaseModel):
    calls: float
    minutes: float
    messages: float
    mb_used: float

@app.get("/")
def get_model_performance():
    return {
        "best_accuracy_score": float(best_accuracy_score),
        "average_target": float(average),
        "model_performance_ratio": float(model_performance),
        "baseline_accuracy": float(baseline_accuracy)
    }

@app.post("/predict")
def predict(behavior: UserBehavior):
    # Convert input to DataFrame
    # Using model_dump() for Pydantic v2
    input_data = pd.DataFrame([behavior.model_dump()])
    prediction = best_model.predict(input_data)
    return {"is_ultra_prediction": int(prediction[0])}
