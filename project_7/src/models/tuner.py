from sklearn.model_selection import RandomizedSearchCV
from src.config import RANDOM_STATE

class HyperparameterTuner:
    """
    Class for tuning hyperparameters using RandomizedSearchCV.
    """
    def __init__(self, model, param_grid, cv=5, scoring='accuracy'):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None

    def tune(self, X_train, y_train):
        """
        Perform randomized search to find the best hyperparameters.
        """
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            random_state=RANDOM_STATE,
            n_iter=10
        )
        random_search.fit(X_train, y_train)
        
        self.best_estimator_ = random_search.best_estimator_
        self.best_score_ = random_search.best_score_
        self.best_params_ = random_search.best_params_
        
        return self.best_estimator_, self.best_score_
