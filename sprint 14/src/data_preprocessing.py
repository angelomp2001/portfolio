# libraries
import pandas as pd

def load_data(file_path):
    df_reviews = pd.read_csv(file_path, sep='\t', dtype={'votes': 'Int64'})
    return df_reviews



