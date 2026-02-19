# main.py
''' Pick the best model for predicting binary classifier with a significant minority ratio, under various compensation strategies. 
compensation strategies: 'balanced weights' logistic regression setting, upsampling, downsampling'''

# libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from main_v2 import *


def define_models(random_state):
    return {
        "Regressions": {
            "LogisticRegression": LogisticRegression(random_state=random_state, solver="liblinear", max_iter=200)
        },
        "Machine Learning": {
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
            "RandomForestClassifier": RandomForestClassifier(random_state=random_state),
        },
    }



# Load data
df = pd.read_csv("data/Churn.csv")
#view(df)

# Clean
handler = DataHandler(df, target_col="Exited")
handler.clean(drop_cols=["RowNumber", "CustomerId", "Surname"])
#see(handler.df)

# Split
data_split = handler.split(split_ratio=(0.6, 0.2, 0.2), random_state=99999)

X_train, X_val, X_test, y_train, y_val, y_test = data_split

X_train, X_val = preprocess_data(X_train, X_val)



