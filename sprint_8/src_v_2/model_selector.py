import pandas as pd
from src_v_2.model_trainer import ModelTrainer

class ModelSelector:
    def __init__(self, model_options: dict):
        self.model_options = model_options
        self.results = pd.DataFrame()

    def run_all(self, data_split):
        X_train, X_val, y_train, y_val = data_split

        for category, models in self.model_options.items():
            print(f"Testing {category} models...")
            for model_name, model in models.items():
                trainer = ModelTrainer(model, model_name)
                trainer.fit(X_train, y_train)
                metrics = trainer.evaluate(X_val, y_val)
                self.results = pd.concat([self.results, pd.DataFrame([metrics])], ignore_index=True)

        return self.results

    def summarize(self):
        summary = self.results.sort_values(by="ROC AUC", ascending=False)
        print("\n=== Summary ===")
        print(summary)
        return summary
