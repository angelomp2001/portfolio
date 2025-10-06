# Model 4 - spaCy, TF-IDF and LGBMClassifier
from lightgbm import LGBMClassifier
from evaluation_model import evaluate_model

def model_4_evaluation(
        model,
        corpus_train,
        corpus_test,
        train_target,
        test_target,
        rows
):
    tf_idf = model
    # set rows to 400 to speed up the training
    rows = rows
    corpus_train_small = corpus_train[:rows]
    corpus_test_small = corpus_test[:rows]
    train_target_small = train_target[:rows]
    test_target_small = test_target[:rows]

    # no lemmatization, just stopwords
    # fit and transform the training data, transform the test data
    tf_idf_train = tf_idf.fit_transform(corpus_train_small)
    tf_idf_test = tf_idf.transform(corpus_test_small)

    # train the model
    lgbm_model = LGBMClassifier(random_state = 12345)
    lgbm_model.fit(tf_idf_train, train_target_small)

    # evaluate the model
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