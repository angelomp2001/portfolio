'''facial recognition model to predict a person's age'''

# libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Modules (files)
# from Interfaces.DataLoader import DataLoader
# from EDA.DataStructure import DataStructureAnalyzer
# from Interfaces.Outputter import OutPut

### Project/Interfaces/DataLoader/
import pandas as pd

class DataLoader:
    @staticmethod # allows you to call a method without creating an instance of the class
    def from_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

### Project/EDA/DataStructure/
# class for exploring structure of data

class DataStructureAnalyzer:
    def __init__(self, data):
        self.data = data
        self.numerical = data.select_dtypes(include=['int64', 'float64']).columns.tolist()  # Use numeric types
        self.categorical = data.select_dtypes(include=['object']).columns.tolist()  # Object types
        self.time = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()  # Datetime types

    def overview(self):
        return {
            'n_rows': self.data.shape[0],
            'n_columns': self.data.shape[1],
            'columns': self.data.columns.tolist(),
            'numerical_columns': self.numerical,
            'categorical_columns': self.categorical,
            'time_columns': self.time,
        }
    
    def dtypes(self):
        return self.data.dtypes.to_dict()
    
    def missing(self):
        return self.data.isnull().sum().to_dict()
    
    def nunique(self):
        return self.data.nunique().to_dict()
    
### Interfaces/Outputter/
import pandas as pd
import matplotlib.pyplot as plt

class OutPut:
    def to_console(self, data:None):
        self.data = data
        """Print data to console."""
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key}: {value}")
                 
        elif isinstance(data, pd.DataFrame | pd.Series):
            print(data.head(5))
            
        else:
            print(f'dict, pd.Dataframe/Series required')

        return self
            
    def view(self):
        """View all columns appropriately based on dtype."""
        for col in self.data.columns:
            dtype = self.data[col].dtype

            print(f"\n--- Viewing column: {col} ({dtype}) ---")

            if pd.api.types.is_numeric_dtype(dtype):
                OutPut._plot_numeric(self.data, col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                OutPut._plot_time(self.data, col)
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                OutPut._plot_categorical(self.data, col)
            else:
                print(f"Unsupported dtype ({dtype}), showing raw values:")
                print(self.data[col].head())

    # --- Private helper methods ---
    @staticmethod
    def _plot_numeric(df, col):
        df[col].plot(kind='hist', bins=30, title=f"{col} (Numeric Distribution)")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def _plot_categorical(df, col):
        value_counts = df[col].value_counts().head(20)
        value_counts.plot(kind='bar', title=f"{col} (Top 20 Categories)")
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def _plot_time(df, col):
        df[col].value_counts().sort_index().plot(kind='line', title=f"{col} (Over Time)")
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()


# Main:
# load data
image_path = r'data/faces/'
labels_path = r'data/faces/labels.csv'
labels = DataLoader.from_csv(labels_path)

# EDA
eda = DataStructureAnalyzer(labels)
output = OutPut()

# save the state of the df at the start
start = eda.overview()

# print missing
output.to_console(eda.missing())

# print dtypes
output.to_console(eda.dtypes())

# print nunique 
output.to_console(eda.nunique())

# print cols as graphs
output.to_console(labels).view()

# labels['real_age'].hist()
# labels.drop('file_name', axis = 1, inplace = True)
# print(labels.head())

