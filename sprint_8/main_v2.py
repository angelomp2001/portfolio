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

    def missing_values(self,
        missing_values_method: str,
        fill_value=0) -> pd.DataFrame:
        """
        missing values handler
        method (str): 'drop', 'fill'
        fill_value: the value to use with the 'fill' method. Can be any type (text, number, etc.). Defaults to 0.
                        
        Returns:
            pd.DataFrame: The DataFrame after handling missing values.
        """
        if missing_values_method == 'drop':
            self.df = self.df.dropna()
        elif missing_values_method == 'fill':
            self.df = self.df.fillna(fill_value)
        else:
            raise ValueError(f"Unknown method: {missing_values_method}")
        return self.df

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

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
        #probs = self.model.predict_proba(X_val)
        preds = self.model.predict(X_val)
        probs = getattr(self.model, "predict_proba", lambda X: None)(X_val)
        roc_auc = roc_auc_score(y_val, probs[:, 1]) if probs is not None else None

        precision, recall, _ = precision_recall_curve(y_val, probs[:, 1]) if probs is not None else (None, None, None)
        pr_auc = auc(recall, precision) if recall is not None else None

        self.results = {
            "Model Name": self.name,
            #"Threshold": threshold,  
            "Accuracy": accuracy_score(y_val, preds),
            "Precision":precision_score(y_val, preds),
            "Recall": recall_score(y_val, preds),
            "F1": f1_score(y_val, preds),
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc
        }
        
        return self.results


# optimization.py
import pandas as pd
import numpy as np
from copy import deepcopy

#from src_v_2.model_trainer import ModelTrainer  # adjust import path as needed

class HyperparameterOptimizer:
    def __init__(self, model, param_grid: dict, metric: str = "ROC AUC", model_name: str = None):
        """
        param_grid example:
        {
            "max_depth": [5, 10, 20],
            "n_estimators": [50, 100]
        }
        """
        self.base_model = model
        self.param_grid = param_grid
        self.metric = metric
        self.model_name = model_name or type(model).__name__
        self.results = pd.DataFrame()
        self.best_params = None
        self.best_score = -np.inf
        self.best_model = None

    def _iter_param_combinations(self):
        """
        Simple generator to iterate over all combinations in param_grid.
        (Equivalent to sklearn's ParameterGrid but minimal.)
        """
        from itertools import product

        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]

        for combo in product(*values): # get very combination of values by key
            params = dict(zip(keys, combo)) # create dict of key: value combos
            yield params # all combinations

    def optimize(self, X_train, y_train, X_val, y_val):
        """
        Train and evaluate model for each param combination.
        """
        for params in self._iter_param_combinations():
            # Create a fresh copy of the model for each run
            model = deepcopy(self.base_model)
            model.set_params(**params)

            trainer = ModelTrainer(model, self.model_name)
            trainer.fit(X_train, y_train)
            metrics = trainer.evaluate(X_val, y_val)
            metrics_with_params = {**metrics, **params, 'Model Name': self.model_name}

            self.results = pd.concat([self.results,
                                      pd.DataFrame([metrics_with_params])],
                                     ignore_index=True)

            # Get score of the optimization metric
            score = metrics.get(self.metric)
            if score is not None and score > self.best_score:
                self.best_score = score
                self.best_params = params
                self.best_model = model  # model is already fitted

        return self.best_params, self.best_score, self.results
    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

def optimize_threshold(model, X_val, y_val, metric: str = "F1"):
    """
    Optimize probability threshold for a binary classifier.
    Assumes model has predict_proba.
    """
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.01, 0.99, 0.02)

    results = []
    best_score = -np.inf
    best_threshold = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        roc_auc = roc_auc_score(y_val, probs)
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, probs)
        pr_auc = auc(recall_curve, precision_curve)

        row = {
            "Threshold": t,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc
        }
        results.append(row)

        score = row[metric]
        if score > best_score:
            best_score = score
            best_threshold = t

    results_df = pd.DataFrame(results)
    return best_threshold, best_score, results_df




# model_selector.py
import pandas as pd
#from src_v_2.model_trainer import ModelTrainer

class ModelSelector:
    def __init__(self, model_options: dict, search_spaces: dict = None, metric: str = "ROC AUC"):
        self.search_spaces = search_spaces or {}
        self.metric = metric
        self.model_options = model_options
        self.results = pd.DataFrame()

    def run_all(self, data_split):
        X_train, X_val, y_train, y_val = data_split

        for category, models in self.model_options.items():
                for model_name, model in models.items():
                    # Check if we have search space for this model
                    param_grid = self.search_spaces.get(model_name)

                    if param_grid:
                        # Hyperparameter optimization
                        optimizer = HyperparameterOptimizer(
                            model=model,
                            param_grid=param_grid,
                            metric=self.metric,
                            model_name=model_name
                        )
                        best_params, best_score, hp_results = optimizer.optimize(
                            X_train, y_train, X_val, y_val
                        )

                        # Log optimization results
                        self.results = pd.concat([self.results, hp_results], ignore_index=True)

                        # Use best model to get final metrics (already trained)
                        best_model = optimizer.best_model
                        # trainer = ModelTrainer(best_model, model_name + " (best)")
                        # trainer.fit(X_train, y_train)  # optional: refit if you want
                        # final_metrics = trainer.evaluate(X_val, y_val)
                        #self.results = pd.concat([self.results,
                        #                        pd.DataFrame([final_metrics])],
                        #                        ignore_index=True)
                    else:
                        # No optimization, just train & evaluate once
                        trainer = ModelTrainer(model, model_name)
                        trainer.fit(X_train, y_train)
                        metrics = trainer.evaluate(X_val, y_val)
                        self.results = pd.concat([self.results,
                                                pd.DataFrame([metrics])],
                                                ignore_index=True)

        return self.results

    def summarize(self):
        summary = self.results.sort_values(by=self.metric, ascending=False)
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

def main():
    # Load data
    df = pd.read_csv("data/Churn.csv")
    random_state = 12345
    model_options = {
        'Regressions': {
            'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)
        },
        'Machine Learning': {
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=random_state),
            'RandomForestClassifier': RandomForestClassifier(random_state=random_state),
        }
    }

    search_spaces = {
        'RandomForestClassifier': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None]
        },
        'DecisionTreeClassifier': {
            'max_depth': [3, 5, 10]
        }
    }
    #view(df)

    def test_case(model_options: dict, search_spaces: dict = None, random_state: int = 12345): 
        #raw logistical regression
        # Clean
        handler = DataHandler(df, target_col="Exited")
        handler.clean(drop_cols=["RowNumber", "CustomerId", "Surname"])
        #see(handler.df)

        handler.missing_values(missing_values_method = 'drop')

        # Split
        data_split = handler.split(split_ratio=(0.6, 0.2, 0.2), random_state=random_state)

        X_train, X_val, X_test, y_train, y_val, y_test = data_split
        X_train, X_val = preprocess_data(X_train, X_val)

        # Models
        selector = ModelSelector(model_options, search_spaces=search_spaces, metric="ROC AUC")

        # Train & Evaluate
        results = selector.run_all(data_split=(X_train, X_val, y_train, y_val))
        summary = selector.summarize()
        print(summary)

    # raw models
    test_case(model_options=model_options, search_spaces=search_spaces, random_state=random_state)
    
    model_options = {
                'Regressions': {
                    'LogisticRegression': LogisticRegression(class_weight= 'balanced', random_state=random_state, solver='liblinear', max_iter=200)}
                }
    
    # logisticRegression(class_weight = 'balanced')
    #test_case(model_options=model_options)


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