import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataHandler:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.features = df.drop(columns=[target_col])
        self.target = df[target_col]

    def clean(self, drop_cols=None):
        """Drop irrelevant columns and duplicates."""
        if drop_cols:
            self.df = self.df.drop(columns=drop_cols, errors='ignore')
        self.df = self.df.drop_duplicates()
        self.features = self.df.drop(columns=[self.target_col])
        self.target = self.df[self.target_col]
        return self

    def split(self, split_ratio=(0.6, 0.2, 0.2), random_state=42):
        """Split into train, validation, and test sets."""
        train_size, val_size, _ = split_ratio
        temp_size = val_size + split_ratio[2]

        X_train, X_temp, y_train, y_temp = train_test_split(
            self.features, self.target, test_size=temp_size, random_state=random_state, stratify=self.target
        )
        relative_val_size = val_size / temp_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-relative_val_size, random_state=random_state, stratify=y_temp
        )

        return (X_train, X_val, X_test, y_train, y_val, y_test)

def preprocess_data(X_train, X_test):
    # Separate categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X_train.select_dtypes(exclude=['object', 'category']).columns

    # --- Scale numerical data ---
    scaler = StandardScaler()
    X_train_num = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]), columns=numeric_cols, index=X_train.index)
    X_test_num = pd.DataFrame(scaler.transform(X_test[numeric_cols]), columns=numeric_cols, index=X_test.index)

    # --- Encode categorical data ---
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]), index=X_train.index)
    X_test_cat = pd.DataFrame(encoder.transform(X_test[categorical_cols]), index=X_test.index)

    # --- Combine numeric + categorical ---
    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

    # set column names to string to avoid issues later
    X_train_final.columns = X_train_final.columns.astype(str)
    X_test_final.columns = X_test_final.columns.astype(str)

    # Fill any remaining NaN values with 0
    X_train_final = X_train_final.fillna(0)
    X_test_final = X_test_final.fillna(0)

    return X_train_final, X_test_final