# Model 9 - BERT
import transformers
import torch
import numpy as np
import math
from tqdm import tqdm
from BERT_text_to_embeddings import BERT_text_to_embeddings
from sklearn.linear_model import LogisticRegression
from evaluation_model import evaluate_model


def model_9_evaluation(
        corpus_train,
        corpus_test,
        train_target,
        test_target,
        rows
):

    # set rows to 4 to speed up the training
    BERT_train_rows = rows

    # get BERT embeddings for the training and test sets
    BERT_enbeddings_train = BERT_text_to_embeddings(corpus_train[:BERT_train_rows], disable_progress_bar=True)
    BERT_enbeddings_test = BERT_text_to_embeddings(corpus_test[:BERT_train_rows], disable_progress_bar=True)

    # train Logistic Regression on the BERT embeddings
    model = LogisticRegression(random_state = 12345)
    model.fit(BERT_enbeddings_train, train_target[:BERT_train_rows])

    # evaluate the model
    model_1 = model
    train_features_1 = BERT_enbeddings_train
    train_target = train_target[:BERT_train_rows]
    test_features_1 = BERT_enbeddings_test
    test_target = test_target[:BERT_train_rows]

    evaluate_model(model_1, train_features_1, train_target, test_features_1, test_target)