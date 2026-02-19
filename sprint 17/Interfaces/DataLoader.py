from libraries import *

# from Interfaces.DataLoader import DataLoader
class DataLoader:
    @staticmethod
    def from_csv(paths: dict) -> dict:
        return {name: pd.read_csv(path) for name, path in paths.items()}
