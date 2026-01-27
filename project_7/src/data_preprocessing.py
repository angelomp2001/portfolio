#libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    return pd.read_csv(file_path)

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
    
    # Set best param to model
    model.set_params(**{param_name: best_param})
    
    return best_param, best_score

#model picker
def model_picker(features, target):
    # Split data
    df = pd.concat([features, target], axis=1)  # Ensure features and target are combined into a single DataFrame
    df_train, df_other = train_test_split(df, test_size=0.4, random_state=12345, stratify=df[target.name])  # training is 60%
    df_valid, df_test = train_test_split(df_other, test_size=0.5, random_state=12345, stratify=df_other[target.name])  # valid & test = .4 * .5 = .2 each
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
    
    # Optimize max_depth for RandomForestClassifier
    rfc_max_depth, _ = hyperparameter_optimizer(
        rfc_model, 'max_depth', 1, 20,
        train_features=train_features, train_target=train_target,
        valid_features=valid_features, valid_target=valid_target, model_name = "RandomForestClassifier"
    )

    # Optimize n_estimators for RandomForestClassifier, while keeping max_depth as the optimal value
    rfc_n_estimators, rfc_best_score = hyperparameter_optimizer(
        rfc_model, 'n_estimators', 10, 100,
        train_features=train_features, train_target=train_target,
        valid_features=valid_features, valid_target=valid_target, model_name = "RandomForestClassifier"
    )
    
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
    
    return best_model, best_model_test_score, train_features, train_target, test_features, test_target

