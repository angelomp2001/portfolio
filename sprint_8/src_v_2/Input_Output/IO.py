import pandas as pd

class Input:
    @staticmethod
    def from_csv(file_path):
        return pd.read_csv(file_path)