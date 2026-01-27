from src.data_preprocessing import *


#import data
path = 'data/users_behavior.csv'
df = load_data(path)

print(df.head())
print(df.describe())

# QC data quality
view(df,'missing values')

# define target and features
target = df['is_ultra']
features = df.drop(target.name,axis = 1)

# select best model
best_model, best_accuracy_score, train_features, train_target, test_features, test_target = model_picker(features, target)

# sanity check using average
average = train_target.mean()

# model performance vs average
model_performance = best_accuracy_score / average
print(f'average:{average}\nmodel_performance:{model_performance}')

# saninty check with DummyClassifier (creates column of target based on strategy and no features)
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)
dummy_clf.fit(train_features, train_target) # it asks for features, but it doesn't use them. 
dummy_y_hat = dummy_clf.predict(test_features)
baseline_accuracy = accuracy_score(test_target, dummy_y_hat)
print("Baseline Accuracy:", baseline_accuracy)

