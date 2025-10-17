import spacy

# load the small English model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# load the small English model
def lemmatization(text):
    
    doc = nlp(text)
    #tokens = [token.lemma_ for token in doc if not token.is_stop]
    lemmas = [token.lemma_ for token in doc]
    
    return ' '.join(lemmas)