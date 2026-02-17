import pandas as pd
import numpy as np

class DataExplorer:
    """
    Class for exploring and summarizing dataframes.
    """
    def __init__(self, dfs):
        """
        Initialize with one or more dataframes.
        
        Args:
            dfs (pd.DataFrame, pd.Series, or dict): Data to explore.
        """
        if isinstance(dfs, pd.DataFrame):
            self.dfs = {'df': dfs}
        elif isinstance(dfs, pd.Series):
            name = dfs.name if dfs.name is not None else 'Series'
            self.dfs = {name: dfs.to_frame()}
        elif isinstance(dfs, dict):
            self.dfs = dfs
        else:
            raise ValueError("Input must be a DataFrame, Series, or dict of DataFrames.")

    def get_summary(self):
        """
        Generate a summary of the dataframes.
        
        Returns:
            pd.DataFrame: A summary dataframe containing counts, missing values, extensive unique values, etc.
        """
        summary_data = []
        for name, df in self.dfs.items():
            for col in df.columns:
                total_count = len(df)
                valid_count = df[col].count()
                missing_count = total_count - valid_count
                missing_percent = (missing_count / total_count) * 100 if total_count > 0 else 0
                
                counts = df[col].value_counts()
                common_values = counts.head(5).index.tolist() if not counts.empty else []
                rare_values = df[col].value_counts(sort=False).head(5).index.tolist() if not counts.empty else []
                
                dtype = df[col].dtype
                
                summary_data.append({
                    'DataFrame': name,
                    'Column': col,
                    'Data Type': dtype,
                    'Total Count': total_count,
                    'Valid Count': valid_count,
                    'Missing Values': f"{missing_count} ({missing_percent:.1f}%)",
                    'Common Values': common_values,
                    'Rare Values': rare_values
                })
        return pd.DataFrame(summary_data)

    def view(self):
        """
        Print the summary to the console.
        """
        print(self.get_summary())
