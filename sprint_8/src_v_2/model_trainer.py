from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc

class ModelTrainer:
    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.results = {}

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_val, y_val):
        """Evaluate with multiple metrics."""
        preds = self.model.predict(X_val)
        probs = getattr(self.model, "predict_proba", lambda X: None)(X_val)
        roc_auc = roc_auc_score(y_val, probs[:, 1]) if probs is not None else None

        precision, recall, _ = precision_recall_curve(y_val, probs[:, 1]) if probs is not None else (None, None, None)
        pr_auc = auc(recall, precision) if recall is not None else None

        self.results = {
            "Model": self.name,
            "Accuracy": accuracy_score(y_val, preds),
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc
        }
        return self.results
