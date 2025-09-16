import pandas as pd

def train_test_split(data_frame: pd.DataFrame, features_list: list, target_column_name: str):

    df_reviews_train = data_frame.query('ds_part == "train"').copy()
    df_reviews_test = data_frame.query('ds_part == "test"').copy()

    train_target = df_reviews_train[target_column_name]
    test_target = df_reviews_test[target_column_name]

    print(df_reviews_train.shape)
    print(df_reviews_test.shape)

    corpus_train = df_reviews_train[features_list]#,'average_rating','rating','sp']]
    corpus_test = df_reviews_test[features_list]#,'average_rating','rating','sp']]

    data_frame.drop('ds_part', axis = 1 , inplace = True)

    return corpus_train, corpus_test, train_target, test_target