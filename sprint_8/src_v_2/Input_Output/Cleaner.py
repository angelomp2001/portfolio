


class Cleaner:
    def __init__(self, raw_data, random_state = 12345):
        self.raw_data = raw_data
        self.df = self.raw_data.copy()
        self.cleaned_data = None
        self.random_state = random_state
        self.n_rows = None
        self.target = None
        self.n_target_minority = None

    def drop(self, col):
        self.df.drop(col, inplace = True, axis = 1)

        return self.df

    def drop_duplicates(self):
        self.df.drop_duplicates(inplace = True)

        return self.df

    def set_rows(self, target, n_rows = [None]):
        self.n_rows = n_rows
        self.target = target
        if n_rows is not None:
            self.df = self.df.sample(n = n_rows, random_state = self.random_state)
            self.df.reset_index(drop = True, inplace = True)

        return self.df

    def set_missing(self, fill_method = 'drop', fill_value = None):
        if fill_method == 'drop':
            self.df.dropna(inplace = True)
        elif fill_method == 'fill':
            self.df.fillna(fill_value, inplace = True)

        return self.df

    



# Cleaner.set_rows(n_rows = n_rows, split_ratio = (0.6, 0.2, 0.2))
