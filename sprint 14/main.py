# Initialization
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm.auto import tqdm

%matplotlib inline
%config InlineBackend.figure_format = 'png'
# the next line provides graphs of better quality on HiDPI screens
%config InlineBackend.figure_format = 'retina'
plt.style.use('seaborn')

# this is to use progress_apply, read more at https://pypi.org/project/tqdm/#pandas-integration
tqdm.pandas()

df_reviews = pd.read_csv('/datasets/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})

# EDA
fig, axs = plt.subplots(2, 1, figsize=(16, 8))

ax = axs[0]

dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates() \
    ['start_year'].value_counts().sort_index()
dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
dft1.plot(kind='bar', ax=ax)
ax.set_title('Number of Movies Over Years')

ax = axs[1]

dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)

dft2.plot(kind='bar', stacked=True, label='#reviews (neg, pos)', ax=ax)

dft2 = df_reviews['start_year'].value_counts().sort_index()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
dft3 = (dft2/dft1).fillna(0)
axt = ax.twinx()
dft3.reset_index(drop=True).rolling(5).mean().plot(color='orange', label='reviews per movie (avg over 5 years)', ax=axt)

lines, labels = axt.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper left')

ax.set_title('Number of Reviews Over Years')

fig.tight_layout()

# Let's check the distribution of number of reviews per movie with the exact counting and KDE (just to learn how it may differ from the exact counting)
fig, axs = plt.subplots(1, 2, figsize=(16, 5))

ax = axs[0]
dft = df_reviews.groupby('tconst')['review'].count() \
    .value_counts() \
    .sort_index()
dft.plot.bar(ax=ax)
ax.set_title('Bar Plot of #Reviews Per Movie')

ax = axs[1]
dft = df_reviews.groupby('tconst')['review'].count()
sns.kdeplot(dft, ax=ax)
ax.set_title('KDE Plot of #Reviews Per Movie')

fig.tight_layout()

df_reviews['pos'].value_counts()

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ax = axs[0]
dft = df_reviews.query('ds_part == "train"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('The train set: distribution of ratings')

ax = axs[1]
dft = df_reviews.query('ds_part == "test"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('The test set: distribution of ratings')

fig.tight_layout()

fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=(2, 1), height_ratios=(1, 1)))

ax = axs[0][0]

dft = df_reviews.query('ds_part == "train"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('The train set: number of reviews of different polarities per year')

ax = axs[0][1]

dft = df_reviews.query('ds_part == "train"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('The train set: distribution of different polarities per movie')

ax = axs[1][0]

dft = df_reviews.query('ds_part == "test"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('The test set: number of reviews of different polarities per year')

ax = axs[1][1]

dft = df_reviews.query('ds_part == "test"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('The test set: distribution of different polarities per movie')

fig.tight_layout()

# Evaluation Procedure
import sklearn.metrics as metrics

def evaluate_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        
        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps
        
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # F1 Score
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'F1 Score') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')        
        ax.set_title(f'ROC Curve')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')        

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))
    
    print(df_eval_stats)
    
    return

# Normalization
# only keep letters and apostrophe
import re
pattern = r"[^a-zA-z\']"
review_norm = []
for row in df_reviews['review']:
    cleaned_text = re.sub(pattern, ' ', row.lower())
    review_norm.append(" ".join(cleaned_text.split()))

df_reviews['review_norm'] = pd.Series(review_norm)

df_reviews['review_norm'].head()

# Train/test split
sp_dict = {'neg': 0, 'pos': 1}
df_reviews['sp'] = df_reviews['sp'].map(sp_dict)
df_reviews['sp'].head()
df_reviews.dropna(inplace = True)

df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

train_target = df_reviews_train['pos']
test_target = df_reviews_test['pos']

print(df_reviews_train.shape)
print(df_reviews_test.shape)

corpus_train = df_reviews_train['review_norm']#,'average_rating','rating','sp']]
corpus_test = df_reviews_test['review_norm']#,'average_rating','rating','sp']]

# Working with models
# model of a constant
from sklearn.dummy import DummyClassifier

df_reviews_train.drop('ds_part', axis = 1 , inplace = True)

test_target_mean = pd.Series(int(train_target.mean()), index =test_target.index)
print(f'accuracy: {np.mean(test_target == test_target_mean)}')

# Model 1 - NLTK, TF-IDF and LR
# Model 1: using stop words
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords

stop_words = list(stopwords.words('english'))
tf_idf = TfidfVectorizer(
    stop_words=stop_words,
    max_features=10000,  # Limit vocabulary size
    min_df=2,           # Ignore rare words
    max_df=0.95         # Ignore too common words
)

tf_idf_train_lr = tf_idf.fit_transform(corpus_train)
tf_idf_test_lr = tf_idf.transform(corpus_test)

lr_model = LogisticRegression(random_state = 12345)
lr_model.fit(tf_idf_train_lr, train_target)


model_1 = lr_model
train_features_1 = tf_idf_train_lr
train_target = train_target
test_features_1 = tf_idf_test_lr
test_target = df_reviews_test['pos']

evaluate_model(model_1, train_features_1, train_target, test_features_1, test_target)
'''
          train  test
Accuracy   0.94  0.88
F1         0.94  0.88
APS        0.98  0.95
ROC AUC    0.98  0.95
'''

# Model 3 - spaCy, TF-IDF and LR
# stopwords and lemmas via spaCy

import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def text_preprocessing_3(text):
    
    doc = nlp(text)
    #tokens = [token.lemma_ for token in doc if not token.is_stop]
    lemmas = [token.lemma_ for token in doc]
    
    return ' '.join(lemmas)

rows = 4000

corpus_train_small = corpus_train[:rows]
corpus_test_small = corpus_test[:rows]
train_target_small = train_target[:rows]
test_target_small = test_target[:rows]

corpus_train_norm = [text_preprocessing_3(text) for text in corpus_train_small]
corpus_test_norm = [text_preprocessing_3(text) for text in corpus_test_small]


tf_idf_train = tf_idf.fit_transform(corpus_train_norm)
tf_idf_test = tf_idf.transform(corpus_test_norm)


lr_model = LogisticRegression(random_state = 12345)
lr_model.fit(tf_idf_train, train_target_small) # could not convert string to float


model_1 = lr_model
train_features_1 = tf_idf_train
train_target_small
test_features_1 = tf_idf_test
test_target_small

evaluate_model(model_1, train_features_1, train_target_small, test_features_1, test_target_small)




'''
rows = 4k  train  test
Accuracy   0.96  0.84
F1         0.96  0.82
APS        0.99  0.92
ROC AUC    0.99  0.92
'''

'''
rows = 4c  train  test
Accuracy   0.99  0.71
F1         0.99  0.61
APS        1.00  0.88
ROC AUC    1.00  0.88
'''

# Model 4 - spaCy, TF-IDF and LGBMClassifier

from lightgbm import LGBMClassifier

rows = 400
corpus_train_small = corpus_train[:rows]
corpus_test_small = corpus_test[:rows]
train_target_small = train_target[:rows]
test_target_small = test_target[:rows]

# no lemmatization
tf_idf_train = tf_idf.fit_transform(corpus_train_small)
tf_idf_test = tf_idf.transform(corpus_test_small)

lgbm_model = LGBMClassifier(random_state = 12345)
lgbm_model.fit(tf_idf_train, train_target_small)


model_1 = lgbm_model
train_features_1 = tf_idf_train
train_target_small
test_features_1 = tf_idf_test
test_target_small

evaluate_model(model_1, train_features_1, train_target_small, test_features_1, test_target_small)
'''
row = 4c/k train  test
Accuracy    1.0  0.70
F1          1.0  0.68
APS         1.0  0.77
ROC AUC     1.0  0.78
'''

# Model 9 - BERT

import torch
import transformers

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
config = transformers.BertConfig.from_pretrained('bert-base-uncased')
model = transformers.BertModel.from_pretrained('bert-base-uncased')

BERT_train_rows = 4

def BERT_text_to_embeddings(texts, max_length=512, batch_size=100, force_device=None, disable_progress_bar=False):
    
    ids_list = []
    attention_mask_list = []

    # text to padded ids of tokens along with their attention masks
    for input_text in texts.iloc[:rows]:
        ids = tokenizer.encode(input_text.lower(), add_special_tokens=True, truncation=True, max_length=max_length)
        padded = np.array(ids + [0]*(max_length - len(ids)))
        attention_mask = np.where(padded != 0, 1, 0)
        ids_list.append(padded)
        attention_mask_list.append(attention_mask)
    
    if force_device is not None:
        device = torch.device(force_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.to(device)
    if not disable_progress_bar:
        print(f'Using the {device} device.')
    
    # gettings embeddings in batches

    embeddings = []

    for i in tqdm(range(math.ceil(len(ids_list)/batch_size)), disable=disable_progress_bar):
            
        ids_batch = torch.LongTensor(ids_list[batch_size*i:batch_size*(i+1)]).to(device)
        attention_mask_batch = torch.LongTensor(attention_mask_list[batch_size*i:batch_size*(i+1)]).to(device)

            
        with torch.no_grad():            
            model.eval()
            batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)   
        embeddings.append(batch_embeddings[0][:,0,:].detach().cpu().numpy())
        
    return np.concatenate(embeddings)

# if you have got the embeddings, it's advisable to save them to have them ready if 
# np.savez_compressed('features_9.npz', train_features_9=train_features_9, test_features_9=test_features_9)

# and load...
# with np.load('features_9.npz') as data:
#     train_features_9 = data['train_features_9']
#     test_features_9 = data['test_features_9']

# my own reviews
# feel free to completely remove these reviews and try your models on your own reviews, those below are just examples
import pandas as pd
import re
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

pattern = r"[^a-zA-Z\']"
cleaned_text_list = []
for text in my_reviews['review']:
    cleaned_text = re.sub(pattern," ",text.lower())
    cleaned_text_list.append(" ".join(cleaned_text.split()))

my_reviews['review_norm'] = pd.Series(cleaned_text_list, index=my_reviews.index)

my_reviews
my_reviews_pos = [0,0,1,0,1,1,0,1]

# Model 2
texts = my_reviews['review_norm']


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf_vectorizer_2 = TfidfVectorizer()

print(f'training size: {corpus_train.shape[0]}')
tf_idf_train = tfidf_vectorizer_2.fit_transform(corpus_train)

model_2 = LogisticRegression(random_state = 12345)
model_2.fit(tf_idf_train, train_target)


my_reviews_pred_prob = model_2.predict_proba(tfidf_vectorizer_2.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{int(round(my_reviews_pred_prob[i]))}:  {review}')

print(f'accuracy: {np.mean(my_reviews_pos == my_reviews_pred_prob)}')

# Model 3
texts = my_reviews['review_norm']


###
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

stop_words = list(stopwords.words('english'))
tfidf_vectorizer_3 = TfidfVectorizer(stop_words=stop_words)

print(f'training size: {corpus_train.shape[0]}')
tf_idf_train = tfidf_vectorizer_3.fit_transform(corpus_train)

model_3 = RandomForestClassifier(random_state = 12345)
model_3.fit(tf_idf_train, train_target)

###



my_reviews_pred_prob = model_3.predict_proba(tfidf_vectorizer_3.transform(texts.apply(lambda x: text_preprocessing_3(x))))[:, 1]


for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{int(round(my_reviews_pred_prob[i]))}:  {review}')

print(f'accuracy: {np.mean(my_reviews_pos == my_reviews_pred_prob)}')

# Model 4
texts = my_reviews['review_norm']

tfidf_vectorizer_4 = tfidf_vectorizer_3

model_4 = LGBMClassifier(random_state = 12345)
model_4.fit(tf_idf_train, train_target)


my_reviews_pred_prob = model_4.predict_proba(tfidf_vectorizer_4.transform(texts.apply(lambda x: text_preprocessing_3(x))))[:, 1]





for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{int(round(my_reviews_pred_prob[i]))}:  {review}')

print(f'accuracy: {np.mean(my_reviews_pos == my_reviews_pred_prob)}')

# Model 9
texts = my_reviews['review_norm']

my_reviews_features_9 = BERT_text_to_embeddings(texts, disable_progress_bar=True)

model_9 = LogisticRegression(random_state = 12345)
model_9.fit(train_features_9, train_target[:BERT_train_rows])

my_reviews_pred_prob = model_9.predict_proba(my_reviews_features_9)[:, 1]


for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{int(round(my_reviews_pred_prob[i]))}:  {review}')

# Conclusions
'''
Here are my results: stopwords + LogisticRegression train test Accuracy 0.94 0.88 F1 0.94 0.88 APS 0.98 0.95 ROC AUC 0.98 0.95

stopwords + lemmatization + logisticregression: rows = 4k train test Accuracy 0.96 0.84 F1 0.96 0.82 APS 0.99 0.92 ROC AUC 0.99 0.92

stopwords + LGBMClassifier row = 4c/k train test Accuracy 1.0 0.70 F1 1.0 0.68 APS 1.0 0.77 ROC AUC 1.0 0.78

lemmatization reduced performance. LGBM overfitted.
'''