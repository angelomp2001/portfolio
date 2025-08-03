
#libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


#import data
df = pd.read_csv('users_behavior.csv')
print(df.head())
print(df.describe())

# QC data quality
def view(dfs, view=None):
    # Convert input to a dictionary of DataFrames if needed
    if isinstance(dfs, pd.DataFrame):
        dfs = {'df': dfs}  # Wrap single DataFrame in a dict with a default name
    elif isinstance(dfs, pd.Series):
        series_name = dfs.name if dfs.name is not None else 'Series'
        dfs = {series_name: dfs.to_frame()}
    else:
        print("Input must be a pandas DataFrame or Series.")
        return

    views = {
        "headers": [],
        "values": [],
        "missing values": [],
        "dtypes": [],
        "summaries": []
    }

    missing_cols = []

    for df_name, df in dfs.items():
        for col in df.columns:
            # Ensure we don't fail on empty columns
            counts = df[col].value_counts()
            common_unique_values = counts.head(5).index.tolist() if not counts.empty else []
            rare_unique_values = df[col].value_counts(sort=False).head(5).index.tolist() if not counts.empty else []
            if df[col].count() > 0:
                data_type = type(df[col].iloc[0])
            else:
                data_type = np.nan

            series_count = df[col].count()
            no_values = len(df) - series_count
            total = no_values + series_count
            no_values_percent = (no_values / total) * 100 if total != 0 else 0

            views["headers"].append({
                'DataFrame': f'{df_name}',
                'Column': col,
                'Common Values': common_unique_values,
            })

            views["values"].append({
                'DataFrame': f'{df_name}',
                'Column': col,
                'Rare Values': rare_unique_values,
            })

            views["missing values"].append({
                'DataFrame': f'{df_name}',
                'Column': col,
                'Series Count': series_count,
                'Missing Values (%)': f'{no_values} ({no_values_percent:.0f}%)'
            })

            views["dtypes"].append({
                'DataFrame': f'{df_name}',
                'Column': col,
                'Common Values': common_unique_values,
                'Data Type': data_type,
            })

            views["summaries"].append({
                'DataFrame': f'{df_name}',
                'Column': col,
                'Common Values': common_unique_values,
                'Rare Values': rare_unique_values,
                'Data Type': data_type,
                'Series Count': series_count,
                'Missing Values': f'{no_values} ({no_values_percent:.0f}%)'
            })

            if no_values > 0:
                missing_cols.append(col)

    code = {
        'headers': "# df.rename(columns={col: col.lower() for col in df.columns}, inplace=True)",
        'values': "# df['column_name'].replace(to_replace='old_value', value=None, inplace=True)\n# df['col_1'] = df['col_1'].fillna('Unknown', inplace=False)",
        'missing values': f"# Check for duplicates or summary statistics\nMissing Columns: {missing_cols}",
        'dtypes': "# df['col'] = df['col'].astype(str) (Int64), (float64) \n# df['col'] = pd.to_datetime(df['col'], format='%Y-%m-%dT%H:%M:%SZ')",
        'summaries': f"DataFrames: {list(dfs.keys())}\ndf.duplicated().sum() \ndf.drop_duplicates() \ndf.duplicated().sum()"
    }

    if view is None or view == "all":
        for view_name, view_data in views.items():
            print(f'{view_name}:\n{pd.DataFrame(view_data)}\n{code.get(view_name, "")}\n')
    elif view in views:
        print(f'{view}:\n{pd.DataFrame(views[view])}\n{code.get(view, "")}\n')
    else:
        print("Invalid view. Available views are: headers, values, dtypes, missing values, summaries, or all.")
            
view(df,'missing values')


# model_picker parameter optimizer:  
def hyperparameter_optimizer(model, param_name, low, high, train_features, train_target, valid_features, valid_target, model_name, tolerance=0.1):
    best_score = -np.inf
    best_param = None
    print(f'Model name:{model_name}\n')
    
    while high - low > tolerance:
        mid = (low + high) / 2
        
        # Set the parameter value
        params = {param_name: int(round(mid))}
        model.set_params(**params)
        
        # Fit the model
        model.fit(train_features, train_target)
        
        # Score the model
        score = model.score(valid_features, valid_target)
        
        # Print current param and score for debugging
        print(f"Param {param_name}: {int(round(mid))}, Score: {score}")
        
        if score > best_score:
            best_score = score
            best_param = int(round(mid))
            low = mid
        else:
            high = mid
    
    return best_param, best_score

#model picker
def model_picker(features, target):
    # Split data
    df = pd.concat([features, target], axis=1)  # Ensure features and target are combined into a single DataFrame
    df_train, df_other = train_test_split(df, test_size=0.4, random_state=12345)  # training is 60%
    df_valid, df_test = train_test_split(df_other, test_size=0.5, random_state=12345)  # valid & test = .4 * .5 = .2 each
    print(f'\
    df_train:{df_train.shape}\n\
    df_valid: {df_valid.shape}\n\
    df_test:{df_test.shape}')
    
    # Define features and targets per df
    train_features = df_train.drop(target.name, axis=1)
    train_target = df_train[target.name]
    valid_features = df_valid.drop(target.name, axis=1)
    valid_target = df_valid[target.name]
    test_features = df_test.drop(target.name, axis=1)
    test_target = df_test[target.name]
    
    # Define base models
    dtc_model = DecisionTreeClassifier(random_state=12345)
    rfc_model = RandomForestClassifier(random_state=12345)
    lr_model = LogisticRegression(random_state=12345, solver='liblinear', max_iter=200)
    
    # Optimize max_depth for DecisionTreeClassifier
    dtc_max_depth, dtc_best_score = hyperparameter_optimizer(
        dtc_model, 'max_depth', 1, 20,
        train_features=train_features, train_target=train_target,
        valid_features=valid_features, valid_target=valid_target, model_name = "DecisionTreeClassifier"
    )
    
    # Optimize max_depth and n_estimators for RandomForestClassifier
    rfc_max_depth, _ = hyperparameter_optimizer(
        rfc_model, 'max_depth', 1, 20,
        train_features=train_features, train_target=train_target,
        valid_features=valid_features, valid_target=valid_target, model_name = "RandomForestClassifier"
    )
    rfc_n_estimators, rfc_best_score = hyperparameter_optimizer(
        rfc_model, 'n_estimators', 10, 100,
        train_features=train_features, train_target=train_target,
        valid_features=valid_features, valid_target=valid_target, model_name = "RandomForestClassifier"
    )
    rfc_model.set_params(max_depth=rfc_max_depth, n_estimators=rfc_n_estimators)
    
    # Fit Logistic Regression model (no hyperparameter optimization in this setup)
    lr_model.fit(train_features, train_target)
    lr_model_score = lr_model.score(valid_features, valid_target)
    
    # Determine the best model based on validation scores
    best_scores = {
        'DecisionTreeClassifier': dtc_best_score,
        'RandomForestClassifier': rfc_best_score,
        'LogisticRegression': lr_model_score
    }
    best_model_name = max(best_scores, key=best_scores.get)
    
    # Retrieve the best model
    if best_model_name == 'DecisionTreeClassifier':
        best_model = dtc_model
        best_model.set_params(max_depth=dtc_max_depth)
    elif best_model_name == 'RandomForestClassifier':
        best_model = rfc_model
    else:
        best_model = lr_model
    
    # Refit the best model on the full training data
    best_model.fit(train_features, train_target)
    
    # Evaluate the best model on the test set
    best_model_test_score = best_model.score(test_features, test_target)
    
    print(f"Best Model: {best_model_name}")
    print(f"Optimal Hyperparameters: {best_model.get_params()}")
    print(f"Test Score: {best_model_test_score}")
    
    return best_model, best_model_test_score

# define target and features
target = df['is_ultra']
features = df.drop(target.name,axis = 1)

# results
best_model, best_score = model_picker(features, target)

#double check
#split data
df = pd.concat([features, target], axis=1) 
df_train, df_other = train_test_split(df, test_size=0.4, random_state=12345, stratify = df['is_ultra']) 
df_valid, df_test = train_test_split(df_other, test_size=0.5, random_state=12345, stratify = df_other['is_ultra'])  

# define train target and features
train_target = df_train['is_ultra']
train_features = df_train.drop(target.name,axis = 1)

# define test target and features
test_target = df_test['is_ultra']
test_features = df_test.drop(test_target.name,axis = 1)

#fit model using optimial hyperparameters found by hyperparameter optimizer
Optimal_Hyperparameters = {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 12, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 78, 'n_jobs': None, 'oob_score': False, 'random_state': 12345, 'verbose': 0, 'warm_start': False}
rfc_model = RandomForestClassifier(**Optimal_Hyperparameters)
rfc_model.fit(train_features,train_target)

#predict y_hat
df_test['y_hat'] = rfc_model.predict(test_features)

# measure accuracy
df_test['error'] = np.where(df_test['y_hat'] != test_target,1,0)

# array
error = np.array(df_test['error'])

# calculate accuracy
accuracy = (len(error) - np.sum(error))/len(error)
print(f'accuracy:{accuracy}')

#sanity check using average
average = df_test['is_ultra'].mean()

# model performance = how much better accuracy is than average, as a multiple.
model_performance = accuracy/average
print(f'average:{average}\nmodel_performance:{model_performance}')

# saninty check with DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)
dummy_clf.fit(train_features, train_target)
df_test['dummy_y_hat'] = dummy_clf.predict(test_features)
baseline_accuracy = accuracy_score(test_target, df_test['dummy_y_hat'])
print("Baseline Accuracy:", baseline_accuracy)

'''
Conclusion:
We wanted to develop a model that would predict 'is_ulta' with the highest possible accuracy, with a threshold for accuracy at 0.75. the target was 30% 1s, and 70% 0s. We needed to take this into account in a two ways:
1. split data accounting for this distribution.
2. test model quality accounting for this distribution.

Splitting training, validation, and test data to account for this distribution was done using the 'stratify' parameter in train_test_split(). The purpose of this parameter is to retain the proportion of values in the target.

Testing model quality took into account the distribution of the target by comparing the model predictions to a naive (no features applied) prediction. This 'dummy' prediction would be the equivalent of guessing the target by just simply keeping the proportions. As a result, it is the equivalent of having no model and makes for a suitable 'baseline' benchmark to measure our 'informed' model against.

The conclusion is that our model outperformed the baseline model by 12% (81% vs 69%). If a 5% outperformance is the threshold, we exceed this threshold by over 2x, suggesting the model was worth building.
'''