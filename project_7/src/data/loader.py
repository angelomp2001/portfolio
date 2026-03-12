import pandas as pd
import os
from src.config import RANDOM_STATE

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    # Handle missing values and duplicates
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    # Sampling
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return df
