# only keep letters and apostrophe
import re
import pandas as pd

def normalization(series: pd.Series):
    pattern = r"[^a-zA-z\']"
    review_norm = []
    for row in series:
        cleaned_text = re.sub(pattern, ' ', row.lower())
        review_norm.append(" ".join(cleaned_text.split()))

    #df[f'{column}_norm'] = pd.Series(review_norm)
    #print(df[f'{column}_norm'].head())

    return pd.Series(review_norm)

