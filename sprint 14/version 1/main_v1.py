# Initialization
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm.auto import tqdm
from normalize_text import normalize_text

from Model_1 import model_1_evaluation
from train_test_split import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from Model_3 import model_3_evaluation
from Model_4 import model_4_evaluation
from Model_9 import model_9_evaluation
from Model_2 import model_2_evaluation
from Model_3_my_reviews import model_3_evaluation_my_reviews
from Model_4_my_reviews import model_4_evaluation_my_reviews
from Model_9_my_reviews import model_9_evaluation_my_reviews

#%matplotlib inline
#%config InlineBackend.figure_format = 'png'
# the next line provides graphs of better quality on HiDPI screens
#%config InlineBackend.figure_format = 'retina'
# print(plt.style.available)
plt.style.use('seaborn-v0_8-pastel')

# this is to use progress_apply, read more at https://pypi.org/project/tqdm/#pandas-integration
tqdm.pandas()

df_reviews = pd.read_csv('sprint 14/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})

# EDA
# Normalization
df_reviews['review_norm'] = normalize_text(series= df_reviews['review']) # df_reviews['review_norm']

# Map sentiment labels to numerical values
sp_dict = {'neg': 0, 'pos': 1}
df_reviews['sp'] = df_reviews['sp'].map(sp_dict)
df_reviews['sp'].head()
df_reviews.dropna(inplace = True)

# Train/test split

corpus_train, corpus_test, train_target, test_target = train_test_split(data_frame=df_reviews, features_list='review_norm', target_column_name = 'pos')

# Working with models
## model of a constant
print(f'mean of train_target')
test_target_mean = pd.Series(int(train_target.mean()), index =test_target.index)
print(f'accuracy: {np.mean(test_target == test_target_mean)}')


# apply stop words to TF-IDF

stop_words = list(stopwords.words('english'))


tf_idf = TfidfVectorizer(
    stop_words=stop_words,
    max_features=10000,  # Limit vocabulary size
    min_df=2,           # Ignore rare words
    max_df=0.95         # Ignore too common words
)

# Model 1 - NLTK, TF-IDF and LR
print('Model 1 - NLTK, TF-IDF and LR')
model_1_evaluation(tf_idf, corpus_train, corpus_test, train_target, test_target)

# Model 3 - spaCy, TF-IDF and LR
print('Model 3 - NLTK, TF-IDF and RF')
model_3_evaluation(tf_idf, corpus_train, corpus_test, train_target, test_target)


# Model 4 - spaCy, TF-IDF and LGBMClassifier
print('Model 4 - NLTK, TF-IDF and LGBMClassifier')
model_4_evaluation(tf_idf,corpus_train, corpus_test, train_target, test_target, rows = 4000)

# Model 9 - BERT
print('Model 9 - BERT and LR')
model_9_evaluation(corpus_train, corpus_test, train_target, test_target, rows = 40)

# my own reviews
print('My reviews')
# feel free to completely remove these reviews and try your models on your own reviews, those below are just examples
my_reviews = pd.DataFrame([
    'I did not simply like it, not my kind of movie.',
    'Well, I was bored and felt asleep in the middle of the movie.',
    'I was really fascinated with the movie',    
    'Even the actors looked really old and disinterested, and they got paid to be in the movie. What a soulless cash grab.',
    'I didn\'t expect the reboot to be so good! Writers really cared about the source material',
    'The movie had its upsides and downsides, but I feel like overall it\'s a decent flick. I could see myself going to see it again.',
    'What a rotten attempt at a comedy. Not a single joke lands, everyone acts annoying and loud, even kids won\'t like this!',
    'Launching on Netflix was a brave move & I really appreciate being able to binge on episode after episode, of this exciting intelligent new drama.'
], columns=['review'])

print(my_reviews.head())
my_reviews_norm = normalize_text(series= my_reviews['review'])

# true sentiment labels for my reviews, just for the sake of having some way to evaluate the models
my_reviews_pos = pd.Series([0,0,1,0,1,1,0,1], name = 'pos') # pd.DataFrame([0,0,1,0,1,1,0,1], columns=['pos'])

# Model 2, custom features - NLTK, TF-IDF and LR
print('Model 2 - NLTK, TF-IDF and LR')
#model_2_evaluation(corpus_train=corpus_train,train_target= train_target,my_reviews= my_reviews['review'],my_reviews_pos= my_reviews_pos)

# Model 3, custom features - NLTK, TF-IDF and RF
print('Model 3 - NLTK, TF-IDF and RF')
#model_3_evaluation_my_reviews(corpus_train, train_target, my_reviews['review'], my_reviews_pos)


# Model 4, custom features - NLTK, TF-IDF and LGBMClassifier
print('Model 4 - NLTK, TF-IDF and LGBMClassifier')
# model_4_evaluation_my_reviews(corpus_train, train_target, my_reviews['review'], my_reviews_pos)


# Model 9, custom features - BERT and LR
print('Model 9 - BERT and LR')
model_9_evaluation_my_reviews(corpus_train, train_target, my_reviews['review'], my_reviews_pos, rows = 400)

# Conclusions
'''
stopwords + LogisticRegression train test Accuracy 0.94 0.88 F1 0.94 0.88 APS 0.98 0.95 ROC AUC 0.98 0.95
stopwords + lemmatization + logisticregression: rows = 4k train test Accuracy 0.96 0.84 F1 0.96 0.82 APS 0.99 0.92 ROC AUC 0.99 0.92
stopwords + LGBMClassifier row = 4c/k train test Accuracy 1.0 0.70 F1 1.0 0.68 APS 1.0 0.77 ROC AUC 1.0 0.78
lemmatization reduced performance. LGBM overfitted.
'''