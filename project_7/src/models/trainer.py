from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """
    def __init__(self, model=None, random_state=12345):
        self.random_state = random_state
        self.model = model

    def set_model(self, model):
        self.model = model

    def train(self, X_train, y_train):
        """
        Train the model.
        """
        if self.model is None:
            raise ValueError("Model is not set.")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.
        
        Returns:
            float: Accuracy score.
        """
        if self.model is None:
             raise ValueError("Model is not set.")
        return self.model.score(X_test, y_test)
    
    def get_params(self):
        return self.model.get_params()

def split_data(df, target_col, test_size=0.4, random_state=12345):
    """
    Split data into train, validation, and test sets.
    
    Returns:
        tuple: (train_features, train_target, valid_features, valid_target, test_features, test_target)
    """
    target = df[target_col]
    features = df.drop(target_col, axis=1)
    
    # Split into train and temp (valid + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    
    # Split temp into valid and test (50% of temp each, so 20% of total each if test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test
