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
        self.numerical = data.select_dtypes(include=['int64', 'float64'])  # Use numeric types
        self.categorical = data.select_dtypes(include=['object'])  # Object types
        self.time = data.select_dtypes(include=['datetime64[ns]'])  # Datetime types

    def overview(self):
        return {
            'n_rows': self.data.shape[0],
            'n_columns': self.data.shape[1],
            'columns': self.data.columns.tolist(),
            'numerical_columns': self.numerical.columns.tolist(),
            'categorical_columns': self.categorical.columns.tolist(),
            'time_columns': self.time.columns.tolist(),
        }
    
    def dtypes(self):
        return self.data.dtypes.to_dict()
    
    def missing(self):
        return self.data.isnull().sum().to_dict()
    
    def nunique(self):
        return self.data.nunique().to_dict()
    
### Interfaces/Outputter/
import pandas as pd

class OutPut:
    @staticmethod
    def to_console(data):
        if isinstance(data, dict):
            # If it's a dictionary, print each key-value pair
            for key, value in data.items():
                print(f"{key}: {value}")
        elif isinstance(data, pd.DataFrame):
            # If it's a DataFrame, use pandas display functionality
            print(data.to_string(index=False))  # Print DataFrame nicely without the index
        else:
            print(f"Unsupported data type ({type(data)})")


# Main:
# load data
image_path = r'data/faces/'
labels_path = r'data/faces/labels.csv'
labels = DataLoader.from_csv(labels_path)

# EDA
eda = DataStructureAnalyzer(labels)

start = eda.overview()

OutPut.to_console(eda.missing())

OutPut.to_console(eda.dtypes())

OutPut.to_console(eda.nunique())

labels['real_age'].hist()
labels.drop('file_name', axis = 1, inplace = True)
print(labels.head())

