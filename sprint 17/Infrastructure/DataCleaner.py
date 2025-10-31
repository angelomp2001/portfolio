from libraries import *


# from infrastructure.DataCleaner import DataCleaner
class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def replace_from_col(self, col: str, value, from_col: str):
        self.df.loc[self.df[col] == value, col] = self.df[from_col]

    def standardize_enddate(self, col: str):
        self.df[col] = self.df[col].where(self.df[col] == 'No', 'Yes')

    def fix_types(self, types: dict):
        for col, dtype in types.items():
            if 'datetime' in dtype:
                self.df[col] = pd.to_datetime(self.df[col])
            else:
                self.df[col] = self.df[col].astype(dtype)
