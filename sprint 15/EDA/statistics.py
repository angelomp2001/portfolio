# get stats on data

class DataStats:
    def __init__(self, data):
        self.data = data

    def numeric_summary(self):
        return self.data.describe()

    def categorical_summary(self):
        return self.data.describe(include='object')

    def correlation_matrix(self):
        return self.data.corr()

    def value_counts(self):
        return {col: self.data[col].value_counts().to_dict() for col in self.data.columns}
    
    