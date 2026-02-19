# Data_handler.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample, shuffle
from typing import Optional

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
        
        self.target = self.df[self.target_col]
        self.features = self.df.drop(columns=[self.target_col])
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

def downsample(
    df: pd.DataFrame,
    target: str = None,
    n_target_majority: Optional[int] = None,
    n_rows: Optional[int] = None,
    random_state: int = 12345,
) -> pd.DataFrame:
    """
    Downsample a DataFrame to address a class imbalance issue or to reduce overall size. 
    Optionally, rows with missing values in the target column can be dropped before processing.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing features and the target column.
        target (str): Name of the target column containing categorical labels (e.g., 0/1).
        n_target_majority (Optional[int]): The majority class will be downsampled to this total.
        n_rows (Optional[int]): Downsample majority to meet this overall size. 
        random_state (int): Random state for reproducibility.
        dropna (bool): If True, drop rows where the target column is NaN before processing.
    
    Returns:
        pd.DataFrame: A new DataFrame with the requested downsampling applied.
    
    """
    if n_target_majority is not None or n_rows is not None:
        # Identify majority (and implicitly minority) using value_counts.
        target_counts = df[target].value_counts()
        # Identify the majority label as the one with the highest count
        majority_label = target_counts.idxmax()
        
        # Split the DataFrame into majority and non-majority (minority) groups.
        df_majority = df[df[target] == majority_label]
        df_minority = df[df[target] != majority_label]
        
        # QC params
        if n_target_majority is not None and n_target_majority >= len(df_majority):
            raise ValueError(
                f"desired_majority ({n_target_majority}) is greater than the current majority count ({len(df_majority)})."
            )
        elif n_rows is not None and n_rows > len(df):
            raise ValueError(f'n_rows larger than df')
        else:
            pass

        #downsample
        if n_target_majority is not None:
            # downsample target majority
            df_majority = resample(
                df_majority,
                replace=False,
                n_samples=int(n_target_majority),
                random_state=random_state
            )
            # Recombine the groups (minority remains unchanged).
            df_downsampled = pd.concat([df_majority, df_minority]).reset_index(drop=True)
        
        elif n_rows is not None and n_target_majority is None:
            #downsample df
            df_downsampled = resample(
                    df,
                    replace=False,
                    n_samples=n_rows,
                    random_state=random_state
                )
        else:
            pass
            
        if n_target_majority is not None and n_rows is not None:
            #downsample df_downsampled to n_rows
            df_downsampled = resample(
                    df_downsampled,
                    replace=False,
                    n_samples=n_rows,
                    random_state=random_state
            )
        else:
            pass
        
        # Shuffle the final DataFrame to mix the rows.
        df_downsampled = shuffle(df_downsampled, random_state=random_state).reset_index(drop=True)
        
        print(f'df_downsampled shape: {df_downsampled.shape}')
        print(f'--- downsample() complete\n')
        return df_downsampled
    
    else:
        print(f'(no downsampling)')
        return df

def upsample(
    df: pd.DataFrame,
    target: str,
    n_target_minority: int | None = None,
    n_rows: int | None = None,
    random_state: int = 12345,
) -> pd.DataFrame:
    """
    Upsample a DataFrame for two possible reasons:
    
    1. To boost the minority class if it is too small (via n_target_minority).
    2. To enlarge the overall DataFrame if the total number of rows is too small (via n_rows).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and the target column.
    target : str
        Name of the target column containing class labels (e.g., 0/1).
    n_target_minority : int, optional
        If provided and greater than the current minority count, the minority class will be
        upsampled (with replacement) to this size.
    n_rows : int, optional
        If provided and greater than the DataFrame's current size (after any minority upsampling),
        the entire DataFrame will be upsampled (with replacement) to this overall number of rows.
    random_state : int, default=12345
        Random state for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame with the requested upsampling applied.
    
    Raises
    ------
    ValueError
        If n_target_minority is less than the current minority count, or
        if n_rows is less than the current DataFrame size (after minority upsampling).
    """
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in DataFrame")

    # Count classes
    target_counts = df[target].value_counts()
    if len(target_counts) < 2:
        raise ValueError("Target column must have at least two classes to define a minority/majority.")

    minority_label = target_counts.idxmin()
    majority_label = target_counts.idxmax()

    df_minority = df[df[target] == minority_label]
    df_majority = df[df[target] == majority_label]

    # 1) Upsample minority class if requested
    if n_target_minority is not None:
        if n_target_minority < len(df_minority):
            raise ValueError(
                f"n_target_minority ({n_target_minority}) is less than the current minority "
                f"count ({len(df_minority)})."
            )
        df_minority = resample(
            df_minority,
            replace=True,
            n_samples=int(n_target_minority),
            random_state=random_state,
        )

    # Recombine
    df_upsampled = pd.concat([df_majority, df_minority], ignore_index=True)

    # 2) Upsample overall DataFrame if requested
    if n_rows is not None:
        if n_rows < len(df_upsampled):
            raise ValueError(
                f"n_rows ({n_rows}) is less than the current total rows ({len(df_upsampled)})."
            )
        df_upsampled = resample(
            df_upsampled,
            replace=True,
            n_samples=int(n_rows),
            random_state=random_state,
        )

    # If any upsampling happened, shuffle once more for good measure
    if n_target_minority is not None or n_rows is not None:
        df_upsampled = shuffle(df_upsampled, random_state=random_state).reset_index(drop=True)
        print(f"df_upsampled shape: {df_upsampled.shape}")
        print("-- upsample() complete")
        return df_upsampled
    else:
        print("(no upsampling)")
        # No changes; return original DataFrame unchanged
        return df

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
        if not hasattr(self.model, "predict_proba"):
            probs = None
        else:
            probs = self.model.predict_proba(X_val)

        if probs is not None:
            roc_auc = roc_auc_score(y_val, probs[:, 1])
            precision, recall, _ = precision_recall_curve(y_val, probs[:, 1])
            pr_auc = auc(recall, precision)
        else:
            roc_auc = None
            pr_auc = None
            precision = recall = None

        self.results = {
            "Model Name": self.name,
            #"Threshold": threshold,  
            "Accuracy": accuracy_score(y_val, preds),
            "Precision":precision_score(y_val, preds, zero_division=0),
            "Recall": recall_score(y_val, preds, zero_division=0),
            "F1": f1_score(y_val, preds, zero_division=0),
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc
        }
        
        return self.results


# optimization.py
import pandas as pd
import numpy as np
from copy import deepcopy

#from src_v_2.model_trainer import ModelTrainer  # adjust import path as needed
def get_actual_max_depth(model):
    """Return the maximum depth actually achieved by the trained model."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    if isinstance(model, RandomForestClassifier):
        # Max depth over all trees in the forest
        return max(est.tree_.max_depth for est in model.estimators_)
    elif isinstance(model, DecisionTreeClassifier):
        return model.tree_.max_depth
    else:
        return None  # For models where depth doesn't apply
    
class HyperparameterOptimizer:
    def __init__(self, model, param_grid: dict, metric: str = "F1", model_name: str = None):
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
            val_results = trainer.evaluate(X_val, y_val)

            # --- get actual max depth after fitting ---
            actual_max_depth = get_actual_max_depth(model)

            # Copy params so we can overwrite max_depth if it was None
            adjusted_params = params.copy()
            if "max_depth" in adjusted_params and adjusted_params["max_depth"] is None:
                # replace None with the actual depth from the fitted model
                adjusted_params["max_depth"] = actual_max_depth

            metrics_with_params = {**val_results, **adjusted_params}

            self.results = pd.concat([
                self.results, pd.DataFrame([metrics_with_params])],
                ignore_index=True)
            
            score = val_results.get(self.metric)
            if score is not None and score > self.best_score:
                self.best_score = score
                self.best_params = params
                self.best_model = model
            
        best_per_score_rows = []
        metrics = self.results.select_dtypes(include=['float64']).columns.tolist()
        for metric in metrics:
            idx_best_value = self.results[metric].idxmax()
            best_row = self.results.loc[idx_best_value]

            # Rebuild a model with the best hyperparameters for this metric
            best_params_for_metric = {hp: best_row[hp] for hp in self.param_grid.keys()}
            model_for_train = deepcopy(self.base_model)
            model_for_train.set_params(**best_params_for_metric)

            trainer_train = ModelTrainer(model_for_train, self.model_name)
            trainer_train.fit(X_train, y_train)

            # Evaluate on TRAIN for the same metric
            train_results = trainer_train.evaluate(X_train, y_train)
            train_score = train_results.get(metric)
            
            row_dict = {
                "Metric Name": metric,
                "Best Val Score": best_row[metric],
                "Train Score": train_score,
                'overfitting_gap': round(best_row[metric] - train_score, ndigits=2),
                "Model Name": best_row["Model Name"],
                **{hp: best_row[hp] for hp in self.param_grid.keys()},
            }

            best_per_score_rows.append(row_dict)

        best_per_score = pd.DataFrame(best_per_score_rows)
        
        return self.best_params, self.best_score, best_per_score #self.results
    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

def threshold_optimizer(model, X_val, y_val, metric: str = "F1"):
    """
    Optimize probability threshold for a binary classifier.
    Assumes model has predict_proba.
    """
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.01, 0.99, 0.02)

    results = []
    best_score = -np.inf
    best_threshold = 0.5

    roc_auc = roc_auc_score(y_val, probs)
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, probs)
    pr_auc = auc(recall_curve, precision_curve)

    for t in thresholds:
        preds = (probs >= t).astype(int)

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)

        row = {
            "Threshold": t,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc,
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
    def __init__(self, model_options: dict, search_spaces: dict = None, metric: str = "F1"):
        self.search_spaces = search_spaces or {}
        self.metric = metric
        self.model_options = model_options
        self.results = pd.DataFrame()

    def run_all(self, data_split, optimize_threshold: bool = False):
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
                        best_params, best_score, best_per_score = optimizer.optimize(
                            X_train, y_train, X_val, y_val
                        )

                        # Log optimization results
                        'currently overwriting results with only latest model score'
                        self.results = best_per_score #pd.concat([self.results, best_per_score], ignore_index=True)

                        print(f'test1 {optimize_threshold}')
                        # --- threshold optimization on the best model ---
                        if optimize_threshold and hasattr(optimizer.best_model, "predict_proba"):
                            best_threshold, best_thr_score, _ = threshold_optimizer(
                                optimizer.best_model, X_val, y_val, metric=self.metric
                            )
                            
                            # attach threshold info to the selected metric row (e.g., ROC AUC)
                            metric_mask = self.results["Metric Name"] == self.metric
                            self.results.loc[metric_mask, "Best Threshold"] = best_threshold
                            self.results.loc[metric_mask, f"Best Threshold Score ({self.metric})"] = best_thr_score

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
        self.results
        # mask = self.results['Metric Name'] == self.metric
        # top = self.results[mask]
        # sorted_results = pd.concat([top, self.results[~mask]])
        return self.results


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

    # Clean
    handler = DataHandler(df, target_col="Exited")
    handler.clean(drop_cols=["RowNumber", "CustomerId", "Surname"])
    #see(handler.df)

    handler.missing_values(missing_values_method = 'drop')

    # Split


    def base_test_case(model_options: dict, search_spaces: dict = None, random_state: int = 12345): 
        #raw logistical regression
        data_split = handler.split(split_ratio=(0.6, 0.2, 0.2), random_state=random_state)

        X_train, X_val, X_test, y_train, y_val, y_test = data_split
        X_train, X_val = preprocess_data(X_train, X_val)

        # Models
        selector = ModelSelector(model_options, search_spaces=search_spaces, metric="F1")

        # Train & Evaluate
        results = selector.run_all(data_split=(X_train, X_val, y_train, y_val))
        print(results)

    # raw models
    base_test_case(model_options=model_options, search_spaces=search_spaces, random_state=random_state)
    
    model_options = {
                'Regressions': {
                    'LogisticRegression': LogisticRegression(class_weight= 'balanced', random_state=random_state, solver='liblinear', max_iter=200)}
                }
    
    # logisticRegression(class_weight = 'balanced')
    base_test_case(model_options=model_options)

    def sampler_analysis(df: pd.DataFrame, target: str, model_options: dict = None, search_spaces: dict = None, optimize_threshold: bool = False,random_state: int = 12345):
        handler_balanced = DataHandler(df, target_col=target)
        data_split_balanced = handler_balanced.split(split_ratio=(0.6, 0.2, 0.2), random_state=random_state)
        X_train_balanced, X_val_balanced, X_test_balanced, y_train_balanced, y_val_balanced, y_test_balanced = data_split_balanced
        X_train_balanced, X_val_balanced = preprocess_data(X_train_balanced, X_val_balanced)
        # Models
        selector_balanced = ModelSelector(model_options or {}, search_spaces=search_spaces, metric='F1')
        # Train & Evaluate
        results_balanced = selector_balanced.run_all(data_split=(X_train_balanced, X_val_balanced, y_train_balanced, y_val_balanced), optimize_threshold=optimize_threshold)
        return results_balanced

    def upsample_case(df: pd.DataFrame, target: str, model_options: dict = None, random_state: int = 12345):
        # upsample
        df_upsampled = upsample(
            df=df,
            target=target,
            n_target_minority=5000,
            n_rows=None,            
            random_state=42,
        )

        results = sampler_analysis(df=df_upsampled, target=target, model_options=model_options, random_state=random_state)
        print(results)
        

    def downsample_case(df: pd.DataFrame, target: str, model_options: dict = None, random_state = random_state):
        # downsample
        df_downsampled = downsample(
            df=df,
            target=target,
            n_target_majority=2000,
            n_rows=None,
            random_state=random_state,
        )
        results = sampler_analysis(df=df_downsampled, target=target, model_options=model_options, random_state=random_state)
        print(results)

    def hyperparameter_grid_search_case(df: pd.DataFrame, target: str, model_options: dict = None, search_spaces: dict = None, optimize_threshold: bool = False, random_state: int = 12345):
        results = sampler_analysis(df=df, target=target, model_options=model_options, search_spaces=search_spaces, optimize_threshold = optimize_threshold, random_state=random_state)
        print(results)
    
    # upsampling case
    model_options = {
        'Regressions': {
            'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)
        },
        'Machine Learning': {
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=random_state),
            'RandomForestClassifier': RandomForestClassifier(random_state=random_state),
        }
    }
    upsample_case(df=df, target="Exited", model_options=model_options, random_state=random_state)

    # downsampling case
    downsample_case(df=df, target="Exited", model_options=model_options, random_state=random_state)

    # hyperparameter grid search case
    hyperparameter_grid_search_case(df=df, target="Exited", model_options=model_options, search_spaces=search_spaces, random_state=random_state)

    # hyperparameter grid search with threshold optimization
    hyperparameter_grid_search_case(df=df, target="Exited", model_options=model_options, search_spaces=search_spaces, optimize_threshold=True, random_state=random_state)





if __name__ == "__main__":
    main()