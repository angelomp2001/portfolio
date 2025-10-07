# libraries
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.model_selection import train_test_split

# libraries for models, excluding tensorflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# libraries for tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# libraries for transforming during cross val and scoring
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Download data
contract_df = pd.read_csv('contract.csv', sep = ',')
internet_df = pd.read_csv('internet.csv', sep = ',')
personal_df = pd.read_csv('personal.csv', sep = ',')
phone_df = pd.read_csv('phone.csv', sep = ',')

dfs = [contract_df, internet_df, personal_df, phone_df]

# Address missing data

# find rows with missing data
for idx, each_df in enumerate(dfs):
    for each_col in each_df:
        print(f'{idx}/{each_col}:{len(each_df[each_col])}')

# Replace missing in TotalCharges with MonthlyCharges
contract_df.loc[contract_df['TotalCharges'] == " ", 'TotalCharges'] = contract_df['MonthlyCharges']

# Address EndData values
# EndData is either the date the user unsubcribed, or "No" if they did not unsubscribe.  
# These are actually two different kinds of information.
# We are only interested in whether or not, regardless of when, so I'll drop date.
# I'll vectorize the column later.
contract_df['EndDate'] = contract_df['EndDate'].where(contract_df['EndDate'] == "No", "Yes")

# Ensure data types are correct



for df in dfs:
    for col in df:
        print(f'{df[col].name}:{df[col][0]}:{df[col].dtype}')


# contract_df['customerID', 'BeginDate', 'EndDate', 'Type', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
# internet_df['customerID', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
# personal_df['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents']
# phone_df['customerID', 'MultipleLines']

# Update BeginDate to datetime
contract_df['BeginDate'] = pd.to_datetime(contract_df['BeginDate'])

# Update EndDate to now be float64
contract_df['TotalCharges'] = contract_df['TotalCharges'].astype('float64')

# Feature engineering
# Extract y/m/d from BeginDate
contract_df['BeginDate_Y'] = contract_df['BeginDate'].dt.year
contract_df['BeginDate_M'] = contract_df['BeginDate'].dt.month
contract_df['BeginDate_D'] = contract_df['BeginDate'].dt.day

# Merge data
for a in dfs[1:]:
    contract_df = contract_df.merge(a, on = 'customerID', how = 'left')

df = contract_df

# define features and target
target = df['EndDate']
features = df.drop([target.name, 'customerID'], axis = 1)

# vectorize target
target = target.replace({"No": 0, "Yes": 1})

# # define column types for vectorization
numerical_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = features.select_dtypes(include=['object']).columns.tolist()

# list of models
list_of_models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GradientBoostingClassifier(),
    lgb.LGBMClassifier(),
    xgb.XGBClassifier(),
    CatBoostClassifier(verbose=0)
]

# vectorize features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# initialize K-fold vars
k_folds = 5
random_state = 12345
kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

# Evaluate each model
for a_model in list_of_models:
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', a_model)])
    scores = cross_validate(pipe, features, target, cv=kf, scoring='roc_auc', return_train_score=True)
    print(f"{a_model.__class__.__name__:<30} Train AUC: {np.mean(scores['train_score']):.3f} | Test AUC: {np.mean(scores['test_score']):.3f}")


# NN model
# initialize vars
drop_rate = 0.0  # scale to fit (0 – 0.5)
train_scores = []
test_scores = []

# cross val score
for train_idx, test_idx in kf.split(features):
    feature_train, feature_test = features.iloc[train_idx], features.iloc[test_idx]
    target_train, target_test = target.iloc[train_idx], target.iloc[test_idx]

    # Preprocess
    X_train = preprocessor.fit_transform(feature_train)
    X_test = preprocessor.transform(feature_test)

    # Build new model for each fold
    nn_model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(drop_rate),
        Dense(256, activation='relu'),
        Dropout(drop_rate),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['AUC'])

    # Fit model
    nn_model.fit(X_train, target_train, epochs=30, batch_size=32, verbose=0)

    # Calc Predictions
    y_pred_train = nn_model.predict(X_train).ravel()
    y_pred_test = nn_model.predict(X_test).ravel()

    # Calc AUC
    train_auc = roc_auc_score(target_train, y_pred_train)
    test_auc = roc_auc_score(target_test, y_pred_test)
    train_scores.append(train_auc)
    test_scores.append(test_auc)

print(f"NeuralNetwork (keras)          Train AUC: {np.mean(train_scores):.3f} | Test AUC: {np.mean(test_scores):.3f}")


# Choose best model based on AUC-ROC

# LogisticRegression             Train AUC: 0.845 | Test AUC: 0.842
# RandomForestClassifier         Train AUC: 1.000 | Test AUC: 0.867
# KNeighborsClassifier           Train AUC: 0.908 | Test AUC: 0.797
# DecisionTreeClassifier         Train AUC: 1.000 | Test AUC: 0.765
# GradientBoostingClassifier     Train AUC: 0.917 | Test AUC: 0.892
# LGBMClassifier                 Train AUC: 0.989 | Test AUC: 0.915
# XGBClassifier                  Train AUC: 1.000 | Test AUC: 0.919
# CatBoostClassifier             Train AUC: 0.980 | Test AUC: 0.920
# NeuralNetwork (keras)          Train AUC: 0.998 | Test AUC: 0.844

'''
In general it looks like the models overfitted, but based on both train and test AUC, I would choose CatBoostClassifier because of the high test AUC and relative distance from train AUC (so you can argue it did not overfit). Although, theoretically, Neural Network (keras) fitness can be scaled to balance underfit and overfit at a desired benchmark by scaling drop_rate.
'''