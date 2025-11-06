


class Cleaner:
    def __init__(self, raw_data, random_state = 12345):
        self.raw_data = raw_data
        self.cleaned_data = None
        self.random_state = random_state
        self.n_rows = None
        self.target = None
        self.n_target_minority = None

    def drop(self, col):
        self.raw_data.drop(col, inplace = True, axis = 1)

    def drop_duplicates(self):
        self.raw_data.drop_duplicates(inplace = True)

    def set_rows(self, n_rows = None, target = None, n_target_minority = None):
        self.n_rows = n_rows
        self.target = target if target is not None else self.target
        self.n_target_minority = n_target_minority



# Cleaner.set_rows(n_rows = n_rows, split_ratio = (0.6, 0.2, 0.2))
