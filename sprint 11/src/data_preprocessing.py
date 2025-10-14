# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from IPython.display import display



def load_and_label_label(path):
    df = pd.read_csv(path)
    df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})

    # define features
    feature_names = ['gender', 'age', 'income', 'family_members']
    features = feature_names

    return df, features

def EDA(df, features):
    ## plot
    g = sns.pairplot(df[features], kind='hist')
    g.figure.set_size_inches(12, 12)

def get_knn(df, features, row, k, metric):
    
    """
    Returns k nearest neighbors

    :param df: pandas DataFrame used to find similar objects within
    :param n: object no for which the nearest neighbours are looked for
    :param k: the number of the nearest neighbours to return
    :param metric: name of distance metric
    """

    nbrs =  sklearn.neighbors.NearestNeighbors(p=metric)
    nbrs.fit(df[features])
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[row][features]], k, return_distance=True)
    
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    return df_res #[index, distance]

def scale_features(df, features):
    # get max abs vals of features array
    transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[features].to_numpy())

    # make copy of df
    df_scaled = df.copy()

    # apply scaler to features of copy df
    df_scaled.loc[:, features] = transformer_mas.transform(df[features].to_numpy())

    return df_scaled

def continuous_to_binary(target):
    target_s = (target > 0).astype(int)
    target = 'insurance_benefits_received'

    # check for the class imbalance with value_counts()
    (target_s).value_counts()

    return target_s, target

# function to eval classifier using f1
def eval_classifier(y_true, y_pred):
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')

# generating random binary outcomes
def rnd_model_predict(P, size, seed=42):
    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)

def f1_scores(thresholds, target):
    for P in thresholds:
        print(f'The probability: {P:.2f}')
        y_pred_rnd =  rnd_model_predict(P=P, size=len(target))
        eval_classifier(target, y_pred_rnd)
        print()

def data_preprocessor(df, features, target, test_size=0.3, random_state=12345):
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=test_size, random_state=random_state)

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

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def f1_knn(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, k=5):
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
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    # calculating r2
    r2_score = np.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')    

def obfuscate_data(df, obfuscate = True, P = None):
    # generate random matrix P with seed
    rng = np.random.default_rng(seed=42)

    if obfuscate:
        # vectorize
        X = df.to_numpy()

        P = rng.random(size=(X.shape[1], X.shape[1]))

        #checking invertibility
        det_P = np.linalg.det(P)
        print("Determinant of P:", det_P)

        # x_ transformed = XP, obfuscate data
        X_transformed = X @ P
        print(X_transformed)
        return X, X_transformed, P
    else:
        # X = x_ transformed / P
        X_recovered = df @ np.linalg.inv(P)
        return X_recovered