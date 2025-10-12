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

# results
best_model, best_score = model_picker(features, target)

#manual double check
#split data
df = pd.concat([features, target], axis=1) 
df_train, df_other = train_test_split(df, test_size=0.4, random_state=12345, stratify = df['is_ultra']) 
df_valid, df_test = train_test_split(df_other, test_size=0.5, random_state=12345, stratify = df_other['is_ultra'])  

# define train target and features
train_target = df_train['is_ultra']
train_features = df_train.drop(target.name,axis = 1)

# define test target and features
test_target = df_test['is_ultra']
test_features = df_test.drop(test_target.name,axis = 1)

#fit model using optimial hyperparameters found by hyperparameter optimizer
Optimal_Hyperparameters = {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 12, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 78, 'n_jobs': None, 'oob_score': False, 'random_state': 12345, 'verbose': 0, 'warm_start': False}
rfc_model = RandomForestClassifier(**Optimal_Hyperparameters)
rfc_model.fit(train_features,train_target)

#predict y_hat
df_test['y_hat'] = rfc_model.predict(test_features)

# measure accuracy
df_test['error'] = np.where(df_test['y_hat'] != test_target,1,0)

# array
error = np.array(df_test['error'])

# calculate accuracy
accuracy = (len(error) - np.sum(error))/len(error)
print(f'accuracy:{accuracy}')

#sanity check using average
average = df_test['is_ultra'].mean()

# model performance = how much better accuracy is than average, as a multiple.
model_performance = accuracy/average
print(f'average:{average}\nmodel_performance:{model_performance}')

# saninty check with DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)
dummy_clf.fit(train_features, train_target)
df_test['dummy_y_hat'] = dummy_clf.predict(test_features)
baseline_accuracy = accuracy_score(test_target, df_test['dummy_y_hat'])
print("Baseline Accuracy:", baseline_accuracy)

'''
Conclusion:
We wanted to develop a model that would predict 'is_ulta' with the highest possible accuracy, with a threshold for accuracy at 0.75. the target was 30% 1s, and 70% 0s. We needed to take this into account in a two ways:
1. split data accounting for this distribution.
2. test model quality accounting for this distribution.

Splitting training, validation, and test data to account for this distribution was done using the 'stratify' parameter in train_test_split(). The purpose of this parameter is to retain the proportion of values in the target.

Testing model quality took into account the distribution of the target by comparing the model predictions to a naive (no features applied) prediction. This 'dummy' prediction would be the equivalent of guessing the target by just simply keeping the proportions. As a result, it is the equivalent of having no model and makes for a suitable 'baseline' benchmark to measure our 'informed' model against.

The conclusion is that our model outperformed the baseline model by 12% (81% vs 69%). If a 5% outperformance is the threshold, we exceed this threshold by over 2x, suggesting the model was worth building.
'''