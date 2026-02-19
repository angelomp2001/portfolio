# Model 2, custom features - NLTK, TF-IDF and LR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from evaluation_model import evaluate_model
import numpy as np

def model_2_evaluation(
        corpus_train,
        train_target,
        my_reviews, # 8
        my_reviews_pos # 8
):
    # TF-IDF vectorizer without stop words
    tfidf_vectorizer_2 = TfidfVectorizer()
    tf_idf_train = tfidf_vectorizer_2.fit_transform(corpus_train)
    tf_idf_test = tfidf_vectorizer_2.transform(my_reviews.to_list())
    
    model_2 = LogisticRegression(random_state = 12345)
    model_2.fit(tf_idf_train, train_target)

    #my_reviews_pred_prob = model_2.predict_proba(tfidf_vectorizer_2.transform(my_reviews))#[:, 1]
    
    # for i, review in enumerate(my_reviews.str.slice(0, 100)):
    #     print(f'{int(round(my_reviews_pred_prob[i]))}:  {review}')

    # print(f'accuracy: {np.mean(my_reviews_pos == my_reviews_pred_prob)}')

    evaluate_model(model_2, tf_idf_train, train_target, tf_idf_test, my_reviews_pos)