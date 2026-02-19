# Model 4, custom features - NLTK, TF-IDF and LGBM
from lightgbm import LGBMClassifier
from lemmatization import lemmatization
from evaluation_model import evaluate_model
import numpy as np
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer

def model_4_evaluation_my_reviews(
        corpus_train,
        train_target,
        my_reviews,
        my_reviews_pos,
):
    # initialize TF-IDF vectorizer
    tfidf_vectorizer_4 = TfidfVectorizer(stop_words=stop_words)
    tf_idf_train = tfidf_vectorizer_4.fit_transform(corpus_train)

    # training size
    model_4 = LGBMClassifier(random_state = 12345)
    model_4.fit(tf_idf_train, train_target)

    # predict probabilities
    # my_reviews_pred_prob = model_4.predict_proba(tfidf_vectorizer_4.transform(my_reviews.apply(lambda x: text_preprocessing(x))))[:, 1]

    # for i, review in enumerate(my_reviews.str.slice(0, 100)):
    #     print(f'{int(round(my_reviews_pred_prob[i]))}:  {review}')

    # print(f'accuracy: {np.mean(my_reviews_pos == my_reviews_pred_prob)}')

    evaluate_model(model_4, tf_idf_train, train_target, tfidf_vectorizer_4.transform(my_reviews.apply(lambda x: lemmatization(x))), my_reviews_pos)