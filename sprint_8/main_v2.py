# Data_handler.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataHandler:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.features = df.drop(columns=[target_col])
        self.target = df[target_col]

    def clean(self, drop_cols=None):
        """Drop irrelevant columns and duplicates."""
        if drop_cols:
            self.df = self.df.drop(columns=drop_cols, errors='ignore')
        self.df = self.df.drop_duplicates()
        self.features = self.df.drop(columns=[self.target_col])
        self.target = self.df[self.target_col]
        return self

    def split(self, split_ratio=(0.6, 0.2, 0.2), random_state=42):
        """Split into train, validation, and test sets."""
        train_size, val_size, _ = split_ratio
        temp_size = val_size + split_ratio[2]

        X_train, X_temp, y_train, y_temp = train_test_split(
            self.features, self.target, test_size=temp_size, random_state=random_state, stratify=self.target
        )
        relative_val_size = val_size / temp_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-relative_val_size, random_state=random_state, stratify=y_temp
        )

        return (X_train, X_val, X_test, y_train, y_val, y_test)

def preprocess_data(X_train, X_test):
    # Separate categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X_train.select_dtypes(exclude=['object', 'category']).columns

    # Scale numerical data
    scaler = StandardScaler()
    X_train_num = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]), columns=numeric_cols, index=X_train.index)
    X_test_num = pd.DataFrame(scaler.transform(X_test[numeric_cols]), columns=numeric_cols, index=X_test.index)

    # Encode categorical data
    X_train_cat = pd.get_dummies(X_train[categorical_cols], prefix=categorical_cols)
    X_test_cat = pd.get_dummies(X_test[categorical_cols], prefix=categorical_cols)
    print(f'X_train_cat: {X_train_cat.columns}')

    # Fill new categories in test with 0 ie drop them. 
    X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

    # Combine numeric + categorical
    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

    # set column names to string
    X_train_final.columns = X_train_final.columns.astype(str)
    X_test_final.columns = X_test_final.columns.astype(str)

    # Fill any remaining NaN values with 0
    X_train_final = X_train_final.fillna(0)
    X_test_final = X_test_final.fillna(0)

    return X_train_final, X_test_final

# model_trainer.py
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

# model_selector.py
import pandas as pd
#from src_v_2.model_trainer import ModelTrainer

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
        return summary


# main.py
''' Pick the best model for predicting binary classifier with a significant minority ratio, under various compensation strategies. 
compensation strategies: 'balanced weights' logistic regression setting, upsampling, downsampling'''

# libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#from src_v_2.data_handler import DataHandler
#from src_v_2.data_handler import preprocess_data
#from src_v_2.model_selector import ModelSelector
#from src_v_2.model_trainer import ModelTrainer

from src.data_explorers import view, see


def define_models(random_state):
    return {
        "Regressions": {
            "LogisticRegression": LogisticRegression(random_state=random_state, solver="liblinear", max_iter=200)
        },
        "Machine Learning": {
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
            "RandomForestClassifier": RandomForestClassifier(random_state=random_state),
        },
    }


def main():
    # Load data
    df = pd.read_csv("data/Churn.csv")
    #view(df)

    # Clean
    handler = DataHandler(df, target_col="Exited")
    handler.clean(drop_cols=["RowNumber", "CustomerId", "Surname"])
    #see(handler.df)

    # Split
    data_split = handler.split(split_ratio=(0.6, 0.2, 0.2), random_state=99999)

    X_train, X_val, X_test, y_train, y_val, y_test = data_split
    X_train, X_val = preprocess_data(X_train, X_val)

    # Models
    models = define_models(random_state=99999)
    selector = ModelSelector(models)

    # Train & Evaluate
    results = selector.run_all((X_train, X_val, y_train, y_val))
    summary = selector.summarize()
    print(summary.head())


if __name__ == "__main__":
    main()

#_________________________

# # balanced logistic regression
# model_options = {
#     'Regressions': {
#         'LogisticRegression': LogisticRegression(random_state=12345, solver='liblinear', class_weight='balanced')
#     }
# }

# print(f'class weight adjustment...')
# best_scores_summary_df, optimized_hyperparameters, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
#     features = features,
#     target = target,
#     n_rows = None,
#     n_target_majority = None,
#     ordinal_cols = None,
#     random_state = random_state,
#     model_options = model_options,
#     split_ratio = (0.6, 0.2, 0.2),
#     missing_values_method= 'drop',
#     fill_value = None,
#     target_threshold = 0.5,
#     metric=metric,
#     target_type='classification'
# )

# lr_balanced_validation_scores = model_scores

# # upsampling
# print(f'upsampling...')
# best_scores_summary_df, optimized_hyperparameters, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
#     features = features,
#     target = target,
#     n_rows = None,
#     n_target_majority = 5000,
#     ordinal_cols = None,
#     random_state = random_state,
#     model_options = None,
#     split_ratio = (0.6, 0.2, 0.2),
#     missing_values_method= 'drop',
#     fill_value = None,
#     target_threshold = 0.5,
#     metric=metric,
#     target_type='classification'
# )
# upsampling_scores = model_scores

# # downsampling

# print(f'downsampling...')
# best_scores_summary_df, optimized_hyperparameters, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
#     features = features,
#     target = target,
#     n_rows = 4000,
#     n_target_majority = .2*10000,
#     ordinal_cols = None,
#     random_state = random_state,
#     model_options = None,
#     split_ratio = (0.6, 0.2, 0.2),
#     missing_values_method= 'drop',
#     fill_value = None,
#     target_threshold = 0.5,
#     target_type='classification',
#     metric=metric
# )
# downsampling_scores = model_scores

# # applying all models but with optimal hyperparameters to test set:
# print(f'testing all models on test data...')
# best_scores_summary_df, optimized_hyperparameters, best_scores_by_model_df, model_scores, _ , model_options = best_model_picker(
#     features = transformed_data[2],
#     target = transformed_data[3],
#     test_features= transformed_data[4],
#     test_target= transformed_data[5],
#     random_state = random_state,
#     model_options= model_options,
#     model_params = optimized_hyperparameters,
#     target_threshold = 0.5,
#     missing_values_method= 'drop',
#     metric=metric,
#     target_type='classification',
# )

# # applying all models but with optimal hyperparameters to test set AND optimized target threshold:
# print(f'testing all models on test data...')
# best_scores_summary_df, optimized_hyperparameters, best_scores_by_model_df, model_scores, transformed_data, model_options = best_model_picker(
#     features = transformed_data[2],
#     target = transformed_data[3],
#     test_features= transformed_data[4],
#     test_target= transformed_data[5],
#     random_state = random_state,
#     model_options= model_options,
#     model_params = optimized_hyperparameters,
#     target_threshold = None,
#     missing_values_method= 'drop',
#     metric=metric,
#     target_type='classification',
# )

# 'Conclusion: RandomForest is generally a superior model.  Downsampling was the best way to maximize Accuracy.'