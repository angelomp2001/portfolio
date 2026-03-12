import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Global configuration
RANDOM_STATE = 12345
TARGET_COL = 'is_ultra'

# Define models to train and evaluate
MODELS = [
    (
        "DecisionTreeClassifier",
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        True, # is_tree
        {'model__max_depth': list(range(1, 21))} # hyperparam_grid (prepended with model__ for pipeline)
    ),
    (
        "RandomForestClassifier",
        RandomForestClassifier(random_state=RANDOM_STATE),
        True, # is_tree
        {'model__max_depth': list(range(1, 21)), 'model__n_estimators': [10, 50, 100]}
    ),
    (
        "LogisticRegression",
        LogisticRegression(random_state=RANDOM_STATE, solver='liblinear', max_iter=200),
        False, # is_tree
        {} # no param grid for this naive run
    )
]
