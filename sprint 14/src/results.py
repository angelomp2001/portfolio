from sklearn.feature_extraction.text import TfidfVectorizer
from src.normalization import normalization
from src.lemmatization import lemmatization
from src.BERT_text_to_embeddings import BERT_text_to_embeddings
from src.evaluation_model import evaluate_model

def results(normalize, lemmatize, stopword, tokenizer, model, features_train, features_test, target_train, target_test, random_state, rows):
    # Apply normalization if specified
    if normalize:
        features_train = normalization(features_train[:rows])
        features_test = normalization(features_test)

    # Apply lemmatization if specified
    if lemmatize:
        features_train = [lemmatization(text) for text in features_train[:rows]]
        features_test = [lemmatization(text) for text in features_test]

    # Tokenization using Tfidf or BERT based on 'tok'
    if tokenizer == 'tf_idf':
        if stopword:
            stopword = 'english'
        else: stopword = None

        tokenizer = TfidfVectorizer(stop_words=stopword)  # Assume 'stop' is a boolean
        tokens_features_train = tokenizer.fit_transform(features_train[:rows])
        tokens_features_test = tokenizer.transform(features_test)
    elif tokenizer == 'BERT':
        tokens_features_train = BERT_text_to_embeddings(features_train[:rows])
        tokens_features_test = BERT_text_to_embeddings(features_test)

    # Model instantiation with the specified predictive model
    model = model(random_state=random_state)
    model.fit(tokens_features_train, target_train[:rows])  # Fit the model with target_train

    # Evaluate the model
    stats = evaluate_model(model, tokens_features_train, target_train[:rows], tokens_features_test, target_test)
    return stats