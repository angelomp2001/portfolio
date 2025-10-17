'''
Objective: test various text models on predicting sentinement (positive vs negative)
two datasets: IMDB reviews and my custom reviews

This script allows for running multiple tests with different combinations of:
- Training and testing datasets
- Text normalization (True/False)
- Lemmatization (True/False)
- Stopword removal (True/False)
- Tokenization method (TF-IDF/BERT)
- Predictive model (Logistic Regression, LGBMClassifier, RandomForestClassifier)
The results of each test are saved in 'output.csv'.
'''

from src.data_preprocessing import *
from data.custom_test_text import *
from src.tests import *
from src.results import *

# load data
path = 'data/imdb_reviews.tsv'
df_reviews = load_data(path)

# set random seed
random_state = 12345


# Train/test split
corpus_train, corpus_test, train_target, test_target = train_test_split(data_frame=df_reviews, features_list='review', target_column_name = 'pos')

# set row for testing purposes.
rows = len(corpus_train)

# parameters that will stay the same between tests
other_params = {
    'features_train': corpus_train,
    'features_test': my_reviews['review'], # corpus_test
    'target_train': train_target,
    'target_test': my_reviews_pos, # test_target
    'random_state': random_state
}

# Test 0
main_params = {
    'normalize': False,
    'lemmatize': False,
    'stopword': False,
    'tokenizer': None,
    'model': 'test_target_mean',
    'rows': rows 
}
test_0_results = test_0(main_params, train_target, test_target)

# Test 1
main_params = {
    'normalize': False,
    'lemmatize': False,
    'stopword': False,
    'tokenizer': 'tf_idf',
    'model': LogisticRegression,
    'rows': rows #len(corpus_train)
}
test_1_results = test_1(main_params, other_params)

# Test 2
main_params = {
        'normalize': True,
        'lemmatize': False,
        'stopword': False,
        'tokenizer': 'tf_idf',
        'model': LogisticRegression,
        'rows': rows #len(corpus_train)
    }

test_2_results = test_2(main_params, other_params)

# Test 3
main_params = {
        'normalize': True,
        'lemmatize': True,
        'stopword': False,
        'tokenizer': 'tf_idf',
        'model': LogisticRegression,
        'rows': rows #len(corpus_train)
    }

test_3_results = test_3(main_params, other_params)

# Test 4
main_params = {
        'normalize': True,
        'lemmatize': True,
        'stopword': True,
        'tokenizer': 'tf_idf',
        'model': LogisticRegression,
        'rows': rows #len(corpus_train)
    }

test_4_results = test_4(main_params, other_params)

# Test 5
main_params = {
        'normalize': False,
        'lemmatize': False,
        'stopword': False,
        'tokenizer': 'BERT',
        'model': LogisticRegression,
        'rows': rows
    }

test_5_results = test_5(main_params, other_params)

# Test 6
main_params = {
        'normalize': True,
        'lemmatize': False,
        'stopword': False,
        'tokenizer': 'BERT',
        'model': LogisticRegression,
        'rows': rows 
    }

test_6_results = test_6(main_params, other_params)

# Test 7
main_params = {
        'normalize': True,
        'lemmatize': True,
        'stopword': False,
        'tokenizer': 'BERT',
        'model': LogisticRegression,
        'rows': rows
    }

test_7_results = test_7(main_params, other_params)

# Test 8
main_params = {
        'normalize': True,
        'lemmatize': True,
        'stopword': True,
        'tokenizer': 'BERT',
        'model': LogisticRegression,
        'rows': rows
    }

test_8_results = test_8(main_params, other_params)

# Test 9
main_params = {
        'normalize': False,
        'lemmatize': False,
        'stopword': False,
        'tokenizer': 'BERT',
        'model': LGBMClassifier,
        'rows': rows
    }

test_9_results = test_9(main_params, other_params)

# Test 10
main_params = {
        'normalize': True,
        'lemmatize': False,
        'stopword': False,
        'tokenizer': 'BERT',
        'model': LGBMClassifier,
        'rows': rows
    }

test_10_results = test_10(main_params, other_params)

# Test 11
main_params = {
        'normalize': True,
        'lemmatize': True,
        'stopword': False,
        'tokenizer': 'BERT',
        'model': LGBMClassifier,
        'rows': rows
    }

test_11_results = test_11(main_params, other_params)

# Test 12
main_params = {
        'normalize': True,
        'lemmatize': True,
        'stopword': True,
        'tokenizer': 'BERT',
        'model': LGBMClassifier,
        'rows': rows
    }

test_12_results = test_12(main_params, other_params)

# Test 13
main_params = {
        'normalize': False,
        'lemmatize': False,
        'stopword': False,
        'tokenizer': 'BERT',
        'model': RandomForestClassifier,
        'rows': rows
    }

test_13_results = test_13(main_params, other_params)

# Test 14
main_params = {
        'normalize': True,
        'lemmatize': False,
        'stopword': False,
        'tokenizer': 'BERT',
        'model': RandomForestClassifier,
        'rows': rows
    }

test_14_results = test_14(main_params, other_params)

# Test 15
main_params = {
        'normalize': True,
        'lemmatize': True,
        'stopword': False,
        'tokenizer': 'BERT',
        'model': RandomForestClassifier,
        'rows': rows
    }

test_15_results = test_15(main_params, other_params)

# Test 16
main_params = {
        'normalize': True,
        'lemmatize': True,
        'stopword': True,
        'tokenizer': 'BERT',
        'model': RandomForestClassifier,
        'rows': rows
    }

test_16_results = test_16(main_params, other_params)