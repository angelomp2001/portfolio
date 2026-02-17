
import pandas as pd
from sklearn.model_selection import train_test_split

# NOTE: inputs function is removed as constants should be defined in main or config.

def load_data(file_paths):
    """
    Loads data from a dictionary of file paths or returns a dictionary of dataframes.
    If input is a list, returns list of DFs.
    If input is a dict {label: path}, returns {label: df}.
    """
    if isinstance(file_paths, dict):
        return {label: pd.read_csv(path) for label, path in file_paths.items()}
    elif isinstance(file_paths, list):
        return [pd.read_csv(path) for path in file_paths]
    else:
        return pd.read_csv(file_paths)

def preprocess_data(dfs):
    """
    Cleans dataframes by dropping duplicates and useless columns like 'id'.
    Expects a dictionary of DataFrames or a single DataFrame.
    """
    processed_dfs = {}
    
    # Iterate if it's a dict, otherwise process single
    if isinstance(dfs, dict):
        iterator = dfs.items()
    elif isinstance(dfs, list):
        iterator = enumerate(dfs)
    else:
        iterator = [('df', dfs)]

    for label, df in iterator:
        # Check for duplicates by ID
        if 'id' in df.columns:
            df = df.drop_duplicates(subset=['id'], keep='first')
            df = df.drop(columns=['id'])
            
        # Add to results
        processed_dfs[label] = df
            
    return processed_dfs