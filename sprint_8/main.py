''' Pick the best model for predicting binary classifier with a significant minority ratio, under various compensation strategies. 
compensation strategies: 'balanced weights' logistic regression setting, upsampling, downsampling'''



from src.best_model_picker import best_model_picker
import pandas as pd

import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Add the common_lib directory to the Python path
sys.path.append("C:\\Users\\Angelo\\Documents\\github\\portfolio\\portfolio")

from src.data_explorers import view, see

# load data
path = 'data/Churn.csv'
df = pd.read_csv(path)

## EDA
#view(df)

" 'I'll keep the header names, encode categorical, ['Exit'] has minority of 20%, which I think is fine. especially out of 10k rows.   "

# columns=['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)
df = df.drop_duplicates()

#see(df)

## data transformation:
# encode categorical
# NOTE: ['Exit'] has minority of 20% and will stay that way:

#define target & identify ordinal categorical vars
target = df['Exited']
features = df.drop(target.name, axis = 1)

random_state = 12345
metric = None
model_options = {
            'Regressions': {
                'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)},
            'Machine Learning': {
                'DecisionTreeClassifier': DecisionTreeClassifier(random_state=random_state),
                'RandomForestClassifier': RandomForestClassifier(random_state=random_state),
                
            }
        }

#raw
print(f'raw...')
best_scores_summary_df, optimized_hyperparameters, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
    features = features,
    target = target,
    n_target_majority = None,
    n_target_minority = None,
    n_rows = None,
    ordinal_cols = None,
    random_state = random_state,
    model_options = model_options,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= 'drop',
    fill_value = None,
    target_threshold = 0.5,
    metric=metric,
    target_type='classification'
)
raw_validation_scores = model_scores

# balanced logistic regression
model_options = {
    'Regressions': {
        'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear', class_weight='balanced')
    }
}

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