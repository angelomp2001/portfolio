from libraries import *

# from infrastructure.DataMerger import DataMerger
class DataMerger:
    @staticmethod
    def merge_all(dfs: dict, on='customerID', how='left') -> pd.DataFrame:
        merged_df = dfs['contract']
        for name, df in dfs.items():
            if name != 'contract':
                merged_df = merged_df.merge(df, on=on, how=how)
        return merged_df
