''' Pick the best model for predicting binary classifier with a significant minority ratio, under various compensation strategies. 
compensation strategies: 'balanced weights' logistic regression setting, upsampling, downsampling'''

# libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src_v_2.data_handler import DataHandler
from src_v_2.data_handler import preprocess_data
from src_v_2.model_selector import ModelSelector
from src_v_2.model_trainer import ModelTrainer

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

    X_train, X_val, X_test, y_train, y_val, y_test = data_split  # however you get them now
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