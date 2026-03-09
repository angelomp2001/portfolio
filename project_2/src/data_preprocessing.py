import pandas as pd
import matplotlib.pyplot as plt

import os

def load_data(path, sample_frac=None, random_state=None):
    df = pd.read_csv(path, sep=';')
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"Sampled {sample_frac*100}% of data from {path} (random state {random_state})")
    return df

def save_statistics(df, prefix, save_dir='data'):
    os.makedirs(save_dir, exist_ok=True)
    stats = df.describe(include='all')
    stats.to_csv(f"{save_dir}/{prefix}_stats.csv")
    print(f"Saved {prefix} statistics to {save_dir}/{prefix}_stats.csv")

def label_categorical(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"Converted '{col}' to categorical.")
    return df

def engineer_datetime_features(df):
    if 'order_dow' in df.columns:
        # Based on data output, 0 and 1 are highest so it's Sunday and Monday.
        # Weekends are commonly 0 (Sunday) and 6 (Saturday)
        df['is_weekend'] = df['order_dow'].isin([0, 6]).astype(int)
        print("Engineered 'is_weekend' feature from 'order_dow'.")
    if 'order_hour_of_day' in df.columns:
        # Create a boolean feature if order is placed during morning hours
        df['is_morning_order'] = ((df['order_hour_of_day'] >= 6) & (df['order_hour_of_day'] < 12)).astype(int)
        print("Engineered 'is_morning_order' feature from 'order_hour_of_day'.")
    return df


def fill_missing(column):
    if column.dtype == 'object':
        column = column.fillna("Unknown")
        print(f"Filled missing values in '{column.name}' with 'Unknown'")
    elif pd.api.types.is_numeric_dtype(column):
        column = column.fillna(999).astype(int)
        print(f"Filled missing values in '{column.name}' with 999 and converted to integer")
    else:
        print(f"No missing values found in '{column.name}'")
    return column



def remove_duplicates(df):
    # Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Handle product_name column
    if 'product_name' in df.columns:
        duplicates = df['product_name'].str.lower().duplicated().sum()
        print(f"Found {duplicates} duplicate product names.")
        
        df['product_name'] = df['product_name'].str.lower()
        df = df.drop_duplicates(subset='product_name', keep='first')
    else:
        print("No 'product_name' column found.")
    
    return df


def verify_data(df):
    for col in ['order_hour_of_day', 'order_dow', 'days_since_prior_order']:
        if col not in df.columns:
            print(f"Column '{col}' not found in data.")
            continue

        column = df[col]
        print(f"\nVerifying '{col}':")
        print("Unique values:", sorted(column.unique()))
        print("Number of unique values:", column.nunique())

        if col == 'order_hour_of_day':
            order_by_hour = column.value_counts().sort_index()
            order_by_hour.plot(kind='bar')
            plt.title('Orders by Hour of Day')
            plt.xlabel('Hour')
            plt.ylabel('Number of Orders')
            plt.show()

        elif col == 'order_dow':
            orders_per_day = column.value_counts().sort_index()
            orders_per_day.plot(kind='bar')
            plt.title('Orders by Day of Week')
            plt.xlabel('Day')
            plt.ylabel('Number of Orders')
            plt.show()
            print(
            '''
            Assuming Sunday = 0, people order most on Sunday and Monday.
            ''')

        elif col == 'days_since_prior_order':
            counts = column.value_counts().sort_index()
            counts.plot(kind='bar')
            plt.title('Days Between Orders')
            plt.xlabel('Days Since Prior Order')
            plt.ylabel('Number of Orders')
            plt.show()
