from libraries import *


# from interfaces.Output import Output
class Output:
    @staticmethod
    def to_console(data):
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key:<30} Train AUC: {value['Train AUC']:.3f} | Test AUC: {value['Test AUC']:.3f}")
        elif isinstance(data, pd.DataFrame):
            print(data.head())
        else:
            print(f"Unsupported data type: {type(data)}")
        return data
