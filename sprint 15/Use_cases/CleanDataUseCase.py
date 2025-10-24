from EDA.transformations import DataTransformer

class CleanDataUseCase:
    def __init__(self, transformer: DataTransformer):
        self.transformer = transformer

    def to_datetime(self, date_columns):
        for col in date_columns:
            self.transformer.to_datetime(col)
        return self.transformer.data
