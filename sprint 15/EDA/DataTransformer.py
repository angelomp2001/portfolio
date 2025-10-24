import pandas as pd

class DataTransformer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def to_datetime(self, column, errors="coerce"):
        """Converts a column to datetime64 safely."""
        self.data[column] = pd.to_datetime(self.data[column], errors=errors)
        return self.data

    def to_numeric(self, column, errors="coerce"):
        """Converts a column to numeric dtype safely."""
        self.data[column] = pd.to_numeric(self.data[column], errors=errors)
        return self.data

    def rename_columns(self, mapping: dict):
        """Renames columns."""
        self.data.rename(columns=mapping, inplace=True)
        return self.data
