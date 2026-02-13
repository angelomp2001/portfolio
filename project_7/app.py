from fastapi import FastAPI
from main import best_accuracy_score, average, model_performance, baseline_accuracy

app = FastAPI()

@app.get("/")
def get_model_performance():
    return {
        "best_accuracy_score": float(best_accuracy_score),
        "average_target": float(average),
        "model_performance_ratio": float(model_performance),
        "baseline_accuracy": float(baseline_accuracy)
    }
