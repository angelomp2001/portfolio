from sklearn.linear_model import LogisticRegression
from evaluation_model import evaluate_model
from text_preprocessing import text_preprocessing


def model_3_evaluation(
        model,
        corpus_train,
        corpus_test,
        train_target,
        test_target
):

    # TF-IDF vectorizer
    tf_idf = model

    # set rows to 4000 to speed up the training
    rows = 4000

    # save new smaller datasets
    corpus_train_small = corpus_train[:rows]
    corpus_test_small = corpus_test[:rows]
    train_target_small = train_target[:rows]
    test_target_small = test_target[:rows]

    # preprocess the text data
    corpus_train_norm = [text_preprocessing(text) for text in corpus_train_small]
    corpus_test_norm = [text_preprocessing(text) for text in corpus_test_small]

    # fit and transform the training data, transform the test data
    tf_idf_train = tf_idf.fit_transform(corpus_train_norm)
    tf_idf_test = tf_idf.transform(corpus_test_norm)

    # train the model
    lr_model = LogisticRegression(random_state = 12345)
    lr_model.fit(tf_idf_train, train_target_small) # could not convert string to float

    # evaluate the model
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