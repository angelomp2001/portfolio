from sklearn.model_selection import GridSearchCV

class HyperparameterTuner:
    """
    Class for tuning hyperparameters using GridSearchCV.
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
        Perform grid search to find the best hyperparameters.
        """
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring
        )
        grid_search.fit(X_train, y_train)
        
        self.best_estimator_ = grid_search.best_estimator_
        self.best_score_ = grid_search.best_score_
        self.best_params_ = grid_search.best_params_
        
        return self.best_estimator_, self.best_score_
