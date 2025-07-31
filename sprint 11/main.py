'''
Objectives:
find leads who are similar to current customers
predict probability of receiving an insurance benefit. 
predict number of insurance benefits likely will receive.
obfuscate data. 
'''

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

# rename cols
df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})
df.sample(10)
df.info()

# we may want to fix the age type (from float to int) though this is not critical

# write your conversion here if you choose:
# it's already as int.  df['age'] = df['age'].astype('int64')

# check to see that the conversion was successful

# now have a look at the data's descriptive statistics. 
df.describe()
# Does everything look okay?

# everything seems ok to me

# EDA
g = sns.pairplot(df, kind='hist')
g.figure.set_size_inches(12, 12)

# Ok, it is a bit difficult to spot obvious groups (clusters) as it is difficult to combine several variables simultaneously (to analyze multivariate distributions). That's where LA and ML can be quite handy.

# Does the data being not scaled affect the kNN algorithm? If so, how does that appear?
# How similar are the results using the Manhattan distance metric (regardless of the scaling)?
feature_names = ['gender', 'age', 'income', 'family_members']
def get_knn(df, row, k, metric):
    
    """
    Returns k nearest neighbors

    :param df: pandas DataFrame used to find similar objects within
    :param n: object no for which the nearest neighbours are looked for
    :param k: the number of the nearest neighbours to return
    :param metric: name of distance metric
    """

    nbrs =  sklearn.neighbors.NearestNeighbors(p=metric)
    nbrs.fit(df[feature_names])
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[row][feature_names]], k, return_distance=True)
    
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    return df_res #[index, distance]

# define features
feature_names = ['gender', 'age', 'income', 'family_members']
features = feature_names

# get max abs vals of features array
transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

# make copy of df
df_scaled = df.copy()

# apply scaler to features of copy df
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())

df_scaled.sample(5)

for idx in df_scaled.sample(5).index:
    print("Unscaled - Euclidean")
    display(get_knn(df, row=idx, k=5, metric=2))

    print("Unscaled - Manhattan")
    display(get_knn(df, row=idx, k=5, metric=1))

    print("Scaled - Euclidean")
    display(get_knn(df_scaled, row=idx, k=5, metric=2))

    print("Scaled - Manhattan")
    display(get_knn(df_scaled, row=idx, k=5, metric=1))

# Does the data being not scaled affect the kNN algorithm? If so, how does that appear?

# Yes, features with larger nnumbers dominate the distance calculation. Here income disproportionatly influences distance.

# How similar are the results using the Manhattan distance metric (regardless of the scaling)?

# The n<=2 nearest neighbors seem to be the same, but then the results differ.

# Build a KNN-based classifier and measure its quality with the F1 metric for k=1..10 for both the original data and the scaled one. That'd be interesting to see how k may influece the evaluation metric, and whether scaling the data makes any difference. You can use a ready implemention of the kNN classification algorithm from scikit-learn (check the link) or use your own.
# Build the dummy model which is just random for this case. It should return "1" with some probability. Let's test the model with four probability values: 0, the probability of paying any insurance benefit, 0.5, 1.
# calculate the target
df['insurance_benefits_received'] = (df['insurance_benefits'] > 0).astype(int)
target = 'insurance_benefits_received'

# check for the class imbalance with value_counts()
(df['insurance_benefits_received']).value_counts()

def eval_classifier(y_true, y_pred):
    
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')
    
# if you have an issue with the following line, restart the kernel and run the notebook again
    #cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    #print('Confusion Matrix')
    #print(cm)

# generating output of a random model

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)

# dummy models and corresponding confusion matrices
for P in [0, df[target].sum() / len(df), 0.5, 1]:

    print(f'The probability: {P:.2f}')
    y_pred_rnd =  rnd_model_predict(P=P, size=len(df))
        
    eval_classifier(df[target], y_pred_rnd)
    
    print()

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

# predict number of benefits
class MyLinearRegression:
    
    def __init__(self):
        
        self.weights = None
    
    def fit(self, X, y):
        
        # adding the unities
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T @ X2) @ X2.T @ y

    def predict(self, X):
        
        # adding the unities
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        y_pred = X2 @ self.weights
        
        return y_pred

def eval_regressor(y_true, y_pred):
    
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')    

# split data:

# redefine target
target = 'insurance_benefits'

# split data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=12345)

# scale features
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

# linear regression

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

#obfuscate data
# define personal info
personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]
X = df_pn.to_numpy()

# Generating a random matrix 
rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))

# check P is invertible
det_P = np.linalg.det(P)
print("Determinant of P:", det_P)

# x_ transformed = XP
X_transformed = X @ P
print(X_transformed)

# test data is recoverable
# X = x_ transformed / P
X_recovered = X_transformed @ np.linalg.inv(P)

#Print all three cases for a few customers
# The original data
# The transformed one
# The reversed (recovered) one
for i in range(3):
    print(f"\nCustomer {i+1}:")
    print("Original:   ", X[i])
    print("Transformed:", X_transformed[i])
    print("Recovered:  ", X_recovered[i])
# These small differences between the original and recovered data are due to floating-point arithmetic precision limitations.

# Test Linear Regression With Data Obfuscation
# random P matrix
X = df[features]
rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))

# ensure matrix is invertible.
det_P = 0
while det_P != 0:
    P = rng.random(size=(X.shape[1], X.shape[1]))
    det_P = np.linalg.det(P)

# Use XP as the new feature matrix
X_obfuscated = X_train @ P
X_test_obfuscated = X_test @ P

# lr for obfuscated

# fit model
lr.fit(X_obfuscated, y_train)
print(lr.weights)

# predict 
y_test_pred = lr.predict(X_test_obfuscated)

# evaluate
eval_regressor(y_test, y_test_pred)


# lr for original data

# fit model
lr.fit(X_train, y_train)
print(lr.weights)

# predict 
y_test_pred = lr.predict(X_test)

# evaluate
eval_regressor(y_test, y_test_pred)

'''
Scale features before using KNN classifier..
Apply any invertible matrix,P, to obfuscate data, but save P!
I think I should go back and refactor my code to work exclusively as matricies
scaling features has no effect on linear regression.
Feature scaling: Impacts KNN, and regularized linear regression models (Ridge, Lasso, and Elastic Net), since these models penalize the coefficients that are affected by scaling. Also necessary if you want to know the impact of a feature on a target. Otherwise, scaling is absorbed by coefficient.
'''