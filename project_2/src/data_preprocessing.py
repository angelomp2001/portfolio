import pandas as pd
import matplotlib.pyplot as plt

def load_data(path):
    return pd.read_csv(path, sep=';')

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
