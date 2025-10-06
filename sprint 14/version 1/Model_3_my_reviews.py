# Model 3, custom features - NLTK, TF-IDF and RF
from text_preprocessing import text_preprocessing
from evaluation_model import evaluate_model
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def model_3_evaluation_my_reviews(
        corpus_train,
        train_target,
        my_reviews,
        my_reviews_pos
):

    ###
    stop_words = list(stopwords.words('english'))
    tfidf_vectorizer_3 = TfidfVectorizer(stop_words=stop_words)

    print(f'training size: {corpus_train.shape[0]}')
    tf_idf_train = tfidf_vectorizer_3.fit_transform(corpus_train)

    model_3 = RandomForestClassifier(random_state = 12345)
    model_3.fit(tf_idf_train, train_target)

    ###
    # my_reviews_pred_prob = model_3.predict_proba(tfidf_vectorizer_3.transform(my_reviews.apply(lambda x: text_preprocessing(x))))[:, 1]


    # for i, review in enumerate(my_reviews.str.slice(0, 100)):
    #     print(f'{int(round(my_reviews_pred_prob[i]))}:  {review}')

    # print(f'accuracy: {np.mean(my_reviews_pos == my_reviews_pred_prob)}')

    evaluate_model(model_3, tf_idf_train, train_target, tfidf_vectorizer_3.transform(my_reviews.apply(lambda x: text_preprocessing(x))) , my_reviews_pos)