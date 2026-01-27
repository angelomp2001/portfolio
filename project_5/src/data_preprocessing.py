import io
import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt



def load_data(path):
    return pd.read_csv(path)

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
        'dtypes': "# df['col'] = df['col'].astype(str)\n# df['col'] = pd.to_datetime(df['col'], format='%Y-%m-%dT%H:%M:%SZ')",
        'summaries': f"DataFrames: {list(dfs.keys())}"
    }

    if view is None or view == "all":
        for view_name, view_data in views.items():
            print(f'{view_name}:\n{pd.DataFrame(view_data)}\n{code.get(view_name, "")}\n')
    elif view in views:
        print(f'{view}:\n{pd.DataFrame(views[view])}\n{code.get(view, "")}\n')
    else:
        print("Invalid view. Available views are: headers, values, dtypes, missing values, summaries, or all.")

def make_lowercase(df):
    for col in df.columns:
        df.rename(columns={col: col.lower()}, inplace=True)
        print(col.lower())

def relabel_missing(df):
    view(df, 'missing values') #6701
    print(f'tbd count: ',(df['user_score'] == 'tbd').sum()) #2424
    df['user_score'] = df['user_score'].replace(to_replace='tbd', value=None)
    print('tbd count:',(df['user_score'] == 'tbd').sum()) #0
    view(df, 'missing values') #9125

def adjust_dtypes(df):
    #user score.  after fixing tbd above, I need to update it
    print("user_score float:")
    df['user_score']= df['user_score'].astype('float64')
    print(df['user_score'].dtype)

    #year of release: should be datetime
    print("year_of_release to year format:")
    df['year_of_release']= pd.to_datetime(df['year_of_release'], format='%Y')
    print(df['year_of_release'].dtype)

    #critic score should be integer
    print("critic score int:")
    #print(df['critic_score'].unique())
    df['critic_score'] = df['critic_score'].astype('Int64')
    print(df['critic_score'].dtype)

def drop_duplicates(df):
    print("remove duplicates:")
    df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    # QC
    df.duplicated().sum()


