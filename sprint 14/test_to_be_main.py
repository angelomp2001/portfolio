import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from train_test_split import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
import numpy as np

from results import results

df_reviews = pd.read_csv('sprint 14/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})

random_state = 12345

# Train/test split
corpus_train, corpus_test, train_target, test_target = train_test_split(data_frame=df_reviews, features_list='review', target_column_name = 'pos')

scores_dict = {'Test':[], 'normalize':[], 'lemmatize':[], 'stopword':[], 'tokenizer':[], 'model':[], 'rows':[], 'F1':[], 'ROC AUC':[], 'APS':[], 'Accuracy':[]}

other_params = {
    'features_train': corpus_train,
    'features_test': corpus_test, 
    'target_train': train_target,
    'target_test': test_target,
    'random_state': random_state
}

# Test 0
scores_dict['Test'].append(0)
main_params = {
    'normalize': False,
    'lemmatize': False,
    'stopword': False,
    'tokenizer': None,
    'model': 'test_target_mean',
    'rows': 4000 #len(corpus_train)
}
# Baseline prediction: predict the mean as probability, and threshold at 0.5 for class labels
baseline_prob = np.full_like(test_target, fill_value=test_target.mean(), dtype=float)
baseline_pred = (baseline_prob >= 0.5).astype(int)

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
scores_dict['F1'].append(round(f1_score(test_target, baseline_pred),2))
scores_dict['ROC AUC'].append(round(roc_auc_score(test_target, baseline_prob),2))
scores_dict['APS'].append(round(average_precision_score(test_target, baseline_prob),2))
scores_dict['Accuracy'].append(round(accuracy_score(test_target, baseline_pred),2))
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

# Test 1
scores_dict['Test'].append(1)
main_params = {
    'normalize': False,
    'lemmatize': False,
    'stopword': False,
    'tokenizer': 'tf_idf',
    'model': LogisticRegression,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

# Test 2
scores_dict['Test'].append(2)
main_params = {
    'normalize': True,
    'lemmatize': False,
    'stopword': False,
    'tokenizer': 'tf_idf',
    'model': LogisticRegression,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

# Test 3
scores_dict['Test'].append(3)
main_params = {
    'normalize': True,
    'lemmatize': True,
    'stopword': False,
    'tokenizer': 'tf_idf',
    'model': LogisticRegression,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

# Test 4
scores_dict['Test'].append(4)
main_params = {
    'normalize': True,
    'lemmatize': True,
    'stopword': True,
    'tokenizer': 'tf_idf',
    'model': LogisticRegression,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

# Test 5
scores_dict['Test'].append(5)
main_params = {
    'normalize': False,
    'lemmatize': False,
    'stopword': False,
    'tokenizer': 'BERT',
    'model': LogisticRegression,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

# Test 6
scores_dict['Test'].append(6)
main_params = {
    'normalize': True,
    'lemmatize': False,
    'stopword': False,
    'tokenizer': 'BERT',
    'model': LogisticRegression,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

# Test 7
scores_dict['Test'].append(7)
main_params = {
    'normalize': True,
    'lemmatize': True,
    'stopword': False,
    'tokenizer': 'BERT',
    'model': LogisticRegression,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

# Test 8
scores_dict['Test'].append(8)
main_params = {
    'normalize': True,
    'lemmatize': True,
    'stopword': True,
    'tokenizer': 'BERT',
    'model': LogisticRegression,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)


# Test 9
scores_dict['Test'].append(9)
main_params = {
    'normalize': False,
    'lemmatize': False,
    'stopword': False,
    'tokenizer': 'BERT',
    'model': LGBMClassifier,
    'rows': 30 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)


# Test 10
scores_dict['Test'].append(10)
main_params = {
    'normalize': True,
    'lemmatize': False,
    'stopword': False,
    'tokenizer': 'BERT',
    'model': LGBMClassifier,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)


# Test 11
scores_dict['Test'].append(11)
main_params = {
    'normalize': True,
    'lemmatize': True,
    'stopword': False,
    'tokenizer': 'BERT',
    'model': LGBMClassifier,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)


# Test 12
scores_dict['Test'].append(12)
main_params = {
    'normalize': True,
    'lemmatize': True,
    'stopword': True,
    'tokenizer': 'BERT',
    'model': LGBMClassifier,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)


# Test 13
scores_dict['Test'].append(13)
main_params = {
    'normalize': False,
    'lemmatize': False,
    'stopword': False,
    'tokenizer': 'BERT',
    'model': RandomForestClassifier,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)


# Test 14
scores_dict['Test'].append(14)
main_params = {
    'normalize': True,
    'lemmatize': False,
    'stopword': False,
    'tokenizer': 'BERT',
    'model': RandomForestClassifier,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)


# Test 15
scores_dict['Test'].append(15)
main_params = {
    'normalize': True,
    'lemmatize': True,
    'stopword': False,
    'tokenizer': 'BERT',
    'model': RandomForestClassifier,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)


# Test 16
scores_dict['Test'].append(16)
main_params = {
    'normalize': True,
    'lemmatize': True,
    'stopword': True,
    'tokenizer': 'BERT',
    'model': RandomForestClassifier,
    'rows': 3000 #len(corpus_train)
}
stats = results(**main_params, **other_params)
stats_dict = stats.to_dict(orient='records')[0]

{scores_dict[key].append(main_params[key]) for key in main_params.keys()}
{scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
scores_dict['model'][-1] = main_params['model'].__name__
print(scores_dict)
scores_df = pd.DataFrame(scores_dict)
scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

