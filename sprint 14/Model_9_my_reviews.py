# Model 9, custom features - BERT and LR
from evaluation_model import evaluate_model
from BERT_text_to_embeddings import BERT_text_to_embeddings
from sklearn.linear_model import LogisticRegression
import numpy as np

def model_9_evaluation_my_reviews(
        corpus_train,
        train_target,
        my_reviews,
        my_reviews_pos,
        rows,
):    
    BERT_train_rows = rows

    # Attention! Running BERT for thousands of texts may take long run on CPU, at least several hours
    train_features_9 = BERT_text_to_embeddings(corpus_train[:BERT_train_rows], force_device=None)

    # to save
    # np.savez_compressed('features_9.npz', train_features_9=train_features_9, test_features_9=test_features_9)

    # to load
    # with np.load('features_9.npz') as data:
    #     train_features_9 = data['train_features_9']
    #     test_features_9 = data['test_features_9']

    my_reviews_features_9 = BERT_text_to_embeddings(my_reviews[:BERT_train_rows], disable_progress_bar=True)

    model_9 = LogisticRegression(random_state = 12345)
    # model_9.fit(train_features_9, train_target[:BERT_train_rows])
    # my_reviews_pred_prob = model_9.predict_proba(my_reviews_features_9)[:, 1]


    # for i, review in enumerate(my_reviews.str.slice(0, 100)):
    #     print(f'{int(round(my_reviews_pred_prob[i]))}:  {review}')


    # print(f'accuracy: {np.mean(my_reviews_pos == my_reviews_pred_prob)}')

    evaluate_model(model_9, train_features_9, train_target, my_reviews_features_9, my_reviews_pos)