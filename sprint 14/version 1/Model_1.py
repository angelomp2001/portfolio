# Model 1 - NLTK, TF-IDF and LR
# Model 1: using stop words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from evaluation_model import evaluate_model

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def model_1_evaluation(
        model,
        corpus_train,
        corpus_test,
        train_target,
        test_target
):

    # get stop words
    # stop_words = list(stopwords.words('english'))

    tf_idf = model

    # fit and transform the training data, transform the test data
    tf_idf_train_lr = tf_idf.fit_transform(corpus_train)
    tf_idf_test_lr = tf_idf.transform(corpus_test)

    # train the model
    lr_model = LogisticRegression(random_state = 12345)
    lr_model.fit(tf_idf_train_lr, train_target)

    # evaluate the model
    model_1 = lr_model
    train_features_1 = tf_idf_train_lr
    train_target = train_target
    test_features_1 = tf_idf_test_lr
    test_target = test_target #df_reviews_test['pos']

    evaluate_model(model_1, train_features_1, train_target, test_features_1, test_target)
    '''
            train  test
    Accuracy   0.94  0.88
    F1         0.94  0.88
    APS        0.98  0.95
    ROC AUC    0.98  0.95
    '''