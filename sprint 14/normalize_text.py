# only keep letters and apostrophe
import re
import pandas as pd

def normalize_text(df: pd.DataFrame, column: str):
    pattern = r"[^a-zA-z\']"
    review_norm = []
    for row in df[column]:
        cleaned_text = re.sub(pattern, ' ', row.lower())
        review_norm.append(" ".join(cleaned_text.split()))

    df[f'{column}_norm'] = pd.Series(review_norm)
    print(df[f'{column}_norm'].head())

