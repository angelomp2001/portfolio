'''
Marketing needs to predict who's likely to be a custom, receive benefits, how many benefits, while protecting their data. 
'''

# libraries
import numpy as np
import pandas as pd
import math
import seaborn as sns
import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from IPython.display import display

# load data
df = pd.read_csv('/datasets/insurance_us.csv')
df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})
df.sample(10)
df.info()
df.describe()

## EDA
g = sns.pairplot(df, kind='hist')
g.figure.set_size_inches(12, 12)

# define features
feature_names = ['gender', 'age', 'income', 'family_members']
features = feature_names

## scale features
# get max abs vals of features array
transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

# make copy of df
df_scaled = df.copy()

# apply scaler to features of copy df
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())

## finding similar records to a random sample of 5 customers
for idx in df_scaled.sample(5).index:
    print("Unscaled - Euclidean")
    display(get_knn(df, row=idx, k=5, metric=2))

    print("Unscaled - Manhattan")
    display(get_knn(df, row=idx, k=5, metric=1))

    print("Scaled - Euclidean")
    display(get_knn(df_scaled, row=idx, k=5, metric=2))

    print("Scaled - Manhattan")
    display(get_knn(df_scaled, row=idx, k=5, metric=1))

## likely to receive benefits
# convert continous number of benefits to binary
df['insurance_benefits_received'] = (df['insurance_benefits'] > 0).astype(int)
target = 'insurance_benefits_received'

# check for the class imbalance with value_counts()
(df['insurance_benefits_received']).value_counts()

# function to eval classifier using f1
def eval_classifier(y_true, y_pred):
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')


# generating random binary outcomes
def rnd_model_predict(P, size, seed=42):
    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)

# dummy outcomes (0, mean, .5, 1) and their corresponding f1 and confusion matrix
for P in [0, df[target].sum() / len(df), 0.5, 1]:
    print(f'The probability: {P:.2f}')
    y_pred_rnd =  rnd_model_predict(P=P, size=len(df))
    eval_classifier(df[target], y_pred_rnd)
    print()

## scale features then test knn scaled vs unscaled
# split data:
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=12345)

# align indices
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# scale features:
# train:

# fit scaler to vectorized features and get max abs vals.
transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(X_train[features].to_numpy())

# copy df to not affect original df
X_train_scaled = X_train.copy()

# copy df to not affect original df
X_test_scaled = X_test.copy()

# apply scaler and save in copied df. 
X_train_scaled.loc[:, features] = transformer_mas.transform(X_train[features].to_numpy())

# apply scaler and save in copied df. 
X_test_scaled.loc[:, features] = transformer_mas.transform(X_test[features].to_numpy())

# test knn unscaled vs scaled 
for k in range(1,11):
    # unscaled
    #print(f'not scaled:')
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred_unscaled = knn.predict(X_test)
    # eval_classifier(y_test, y_pred_unscaled)
    print(f'k:{k}')

    #scaled
    #print(f'scaled:')
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_scaled, y_train)
    y_pred_scaled = knn.predict(X_test_scaled)
    
    eval_classifier(y_test, y_pred_unscaled)
    eval_classifier(y_test, y_pred_scaled)
    #print(f'k:{k}')

# results
# k=2
# F1: 0.64
# Confusion Matrix
# [[0.866      0.01942857]
#  [0.05085714 0.06371429]]
# F1: 0.94
# Confusion Matrix
# [[0.88171429 0.00371429]
#  [0.00885714 0.10571429]]

# k=3
# F1: 0.39
# Confusion Matrix
# [[0.88257143 0.00285714]
#  [0.08571429 0.02885714]]
# F1: 0.91
# Confusion Matrix
# [[0.884      0.00142857]
#  [0.01857143 0.096     ]]

# k=4
# F1: 0.41
# Confusion Matrix
# [[0.87285714 0.01257143]
#  [0.082      0.03257143]]
# F1: 0.93
# Confusion Matrix
# [[0.88171429 0.00371429]
#  [0.01142857 0.10314286]]

# k=5
# F1: 0.19
# Confusion Matrix
# [[0.88228571 0.00314286]
#  [0.10257143 0.012     ]]
# F1: 0.91
# Confusion Matrix
# [[0.88371429 0.00171429]
#  [0.01828571 0.09628571]]

# k=6
# F1: 0.19
# Confusion Matrix
# [[0.88       0.00542857]
#  [0.10171429 0.01285714]]
# F1: 0.93
# Confusion Matrix
# [[0.88257143 0.00285714]
#  [0.01314286 0.10142857]]

# k=7
# F1: 0.06
# Confusion Matrix
# [[0.88428571 0.00114286]
#  [0.11114286 0.00342857]]
# F1: 0.90
# Confusion Matrix
# [[0.88428571 0.00114286]
#  [0.01971429 0.09485714]]

# k=8
# F1: 0.07
# Confusion Matrix
# [[0.88314286 0.00228571]
#  [0.11057143 0.004     ]]
# F1: 0.93
# Confusion Matrix
# [[0.88371429 0.00171429]
#  [0.01371429 0.10085714]]

# k=9
# F1: 0.00
# Confusion Matrix
# [[0.88542857 0.        ]
#  [0.11457143 0.        ]]
# F1: 0.90
# Confusion Matrix
# [[0.88428571 0.00114286]
#  [0.01971429 0.09485714]]

# k=10
# F1: 0.00
# Confusion Matrix
# [[8.85142857e-01 2.85714286e-04]
#  [1.14285714e-01 2.85714286e-04]]
# F1: 0.93
# Confusion Matrix
# [[0.88314286 0.00228571]
#  [0.01371429 0.10085714]]

## predict number of benefits scaled vs unscaled
# make model
class MyLinearRegression:
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        # adding constant
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)

        # calculating weights
        self.weights = np.linalg.inv(X2.T @ X2) @ X2.T @ y

    def predict(self, X):
        # adding constant
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        
        # predicting y
        y_pred = X2 @ self.weights
        return y_pred
    
def eval_regressor(y_true, y_pred):
    # calculating RMSE
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    # calculating r2
    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')    

# prepare data
# redefine target
target = 'insurance_benefits'

# split data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=12345)

# scale features
# fit scaler to vectorized features and get max abs vals.
transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(X_train[features].to_numpy())

# copy df to not affect original df
X_train_scaled = X_train.copy()

# copy df to not affect original df
X_test_scaled = X_test.copy()

# apply scaler and save in copied df. 
X_train_scaled.loc[:, features] = transformer_mas.transform(X_train[features].to_numpy())

# apply scaler and save in copied df. 
X_test_scaled.loc[:, features] = transformer_mas.transform(X_test[features].to_numpy())

# Apply linear regression
# initialize class
lr = MyLinearRegression()

# not scaled:
# fit model
lr.fit(X_train, y_train)
print(lr.weights)

# predict 
y_test_pred = lr.predict(X_test)

# evaluate
eval_regressor(y_test, y_test_pred)

#scaled
# fit model
lr.fit(X_train_scaled, y_train)
print(lr.weights)

# predict 
y_test_pred = lr.predict(X_test_scaled)

# evaluate
eval_regressor(y_test, y_test_pred)

'''
same results between scaled and unscaled data. 
'''

## obfuscating data
personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]

# vectorize
X = df_pn.to_numpy()

# generate random matrix P with seed
rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))

#checking invertibility
det_P = np.linalg.det(P)
print("Determinant of P:", det_P)

# x_ transformed = XP, obfuscate data
X_transformed = X @ P
print(X_transformed)

# recover obfuscated data
# X = x_ transformed / P
X_recovered = X_transformed @ np.linalg.inv(P)

# 3 tests
for i in range(3):
    print(f"\nCustomer {i+1}:")
    print("Original:   ", X[i])
    print("Transformed:", X_transformed[i])
    print("Recovered:  ", X_recovered[i])

# testing linear regression with obfuscated data
# set X and P
X = df[features]
rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))

# ensure P is invertible.
det_P = 0
while det_P != 0:
    P = rng.random(size=(X.shape[1], X.shape[1]))
    det_P = np.linalg.det(P)

# Use XP as the new feature matrix
X_obfuscated = X_train @ P
X_test_obfuscated = X_test @ P

## lr for obfuscated
# fit model
lr.fit(X_obfuscated, y_train)
print(lr.weights)

# predict 
y_test_pred = lr.predict(X_test_obfuscated)

# evaluate
eval_regressor(y_test, y_test_pred)


## lr for original data
# fit model
lr.fit(X_train, y_train)
print(lr.weights)

# predict 
y_test_pred = lr.predict(X_test)

# evaluate
eval_regressor(y_test, y_test_pred)

'''
technical conclusions:
Scale features before using KNN classifier. Scaling features has no effect on linear regression.
Apply any invertible matrix,P, to obfuscate data, but save P!
''' 