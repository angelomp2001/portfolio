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
        probs = self.model.predict_proba(X_val)
        preds = self.model.predict(X_val)
        #probs = getattr(self.model, "predict_proba", lambda X: None)(X_val)
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
    
    def optimizer(self, param_to_optimize: None, metric_to_use: None):
        self.param_to_optimize = param_to_optimize
        self.metric_to_use = metric_to_use

        #Apply any OTHER model params
        provided_model_params = self.model.get(model_name).get_params().copy()
        current_model_params = self.model.get_params().copy()
        current_model_params.update(provided_model_params)
        

        "if metric is not provided"
        if metric is None:
            metric = metrics
        elif not isinstance(metric, list):
            metric = [metric]

        "optimize a hyperparameter"
        # Algo for hyperparameter optimization
        threshold = 0.5 #keep constant while optimizing hyperparameters

        for col in metric:

            while high - low > tolerance:
                
                # get parameter value
                mid = (low + high) / 2
                
                # save the parameter value
                params = {param_to_optimize: int(round(mid))}
                
                # update model params
                current_model_params.update(params)

                #update model with params
                model.set_params(**current_model_params)
                
                #Re/Fit model
                model.fit(train_features, train_target)
                
                # Predict target
                y_pred = model.predict_proba(score_features)
                
                # score prediction
                accuracy, precision, recall, f1 = categorical_scorer(
                    target=score_target,
                    y_pred=y_pred[:, 1],
                    threshold=threshold 
                )

                #AUC values
                roc_auc = roc_auc_score(score_target, y_pred[:, 1])
                pr_auc = average_precision_score(score_target, y_pred[:, 1])

                # log this iteration
                row_data = {
                    "Model Name": model_name,
                    "Parameter Name": param_to_optimize,
                    "Threshold": threshold, # keep constant b/c we are optimizing hyperparameters"
                    "Parameter": mid, 
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "ROC AUC": roc_auc,
                    "PR AUC": pr_auc
                }

                row_values = pd.DataFrame([row_data])
                hyperparameter_table = pd.concat([hyperparameter_table, row_values], ignore_index=True)
                
                # Choose metric for optimization
                score = row_values[col].iloc[0]
                
                # update best score
                if score > best_score:
                    best_score = score

                    # update model param with param of best score
                    params = {param_to_optimize: int(round(mid))}
                    

                    #set new low for next iteration
                    low = mid + tolerance
                    
                else:
                    # set new high for next iteration
                    high = mid - tolerance            

        "Return optimized param"
        # get optimized parameter
        optimized_param = current_model_params.get(param_to_optimize)

        if metric is None or len(metric) > 1:
            pass
            # return multiple metrics
            #return optimized_param, current_model_params, metrics
        
        else:
            # get best scores of desired metric
            best_scores = hyperparameter_table.loc[hyperparameter_table[col] == best_score]

            # save best scores of desired metric to output
            hyperparameter_table = pd.concat([hyperparameter_table, best_scores], ignore_index=True)

            # return best model params, their scores, and the whole scores df
            #return optimized_param, best_score, metrics




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

def main():
    # Load data
    df = pd.read_csv("data/Churn.csv")
    random_state = 12345
    model_options = {
                'Regressions': {
                    'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)},
                'Machine Learning': {
                    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=random_state),
                    'RandomForestClassifier': RandomForestClassifier(random_state=random_state),
                    
                }
            }
    #view(df)

    def test_case(model_options = model_options): 
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
        selector = ModelSelector(model_options)

        # Train & Evaluate
        results = selector.run_all((X_train, X_val, y_train, y_val))
        summary = selector.summarize()
        print(summary.head())

    # raw models
    test_case(model_options=model_options)
    
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