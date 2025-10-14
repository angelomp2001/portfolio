'''
Marketing needs to predict who's likely to be a custom, receive benefits, how many benefits, while protecting their data. 
'''

from src.data_preprocessing import *


# load data
path = 'data/insurance_us.csv'
df, features = load_and_label_label(path)
df.sample(10)
df.info()
df.describe()

# EDA
EDA(df, features)

## scale features
df_scaled = scale_features(df, features)

## finding similar records to a random sample of 5 customers
for idx in df_scaled.sample(5).index:
    print("Unscaled - Euclidean")
    display(get_knn(df, features, row=idx, k=5, metric=2))

    print("Unscaled - Manhattan")
    display(get_knn(df, features, row=idx, k=5, metric=1))

    print("Scaled - Euclidean")
    display(get_knn(df_scaled, features, row=idx, k=5, metric=2))

    print("Scaled - Manhattan")
    display(get_knn(df_scaled, features, row=idx, k=5, metric=1))

## likely to receive benefits (target label)
# convert continous number of benefits to binary
target_s, target = continuous_to_binary(df['insurance_benefits'])
df['insurance_benefits'] = target_s

# dummy thresholds (0, mean, .5, 1) and their corresponding f1 and confusion matrix
f1_scores([0, target_s.sum() / len(df), 0.5, 1], target_s)

## scale features then test knn scaled vs unscaled
# split data:
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = data_preprocessor(df, features, target_s.name, test_size=0.3, random_state=12345)


# test knn unscaled vs scaled
f1_knn(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, k=5)

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

X, X_transformed, P = obfuscate_data(df_pn, obfuscate = True)

# recover obfuscated data
X_recovered = obfuscate_data(X_transformed, obfuscate = False, P = P)

# 3 tests to measure impact of obfuscation data on regression
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