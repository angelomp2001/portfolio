from libraries import *

# from infrastructure.FeatureEngineer import FeatureEngineer
class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def extract_date_parts(self, col: str):
        self.df['Year'] = self.df[col].dt.year
        self.df['Month'] = self.df[col].dt.month
        self.df['Day'] = self.df[col].dt.day

