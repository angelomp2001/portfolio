import pandas as pd
import numpy as np
import pytest
from src.data_preprocessing import fill_missing, remove_duplicates, label_categorical, engineer_datetime_features

def test_fill_missing_categorical():
    df = pd.DataFrame({'col': ['A', np.nan, 'C']})
    filled_col = fill_missing(df['col'])
    assert filled_col.isna().sum() == 0
    assert 'Unknown' in filled_col.values

def test_fill_missing_numeric():
    df = pd.DataFrame({'col': [1.0, np.nan, 3.0]})
    filled_col = fill_missing(df['col'])
    assert filled_col.isna().sum() == 0
    assert 999 in filled_col.values

def test_remove_duplicates():
    df = pd.DataFrame({'id': [1, 2, 2], 'product_name': ['A', 'b', 'B']})
    clean_df = remove_duplicates(df)
    assert len(clean_df) == 2

def test_label_categorical():
    df = pd.DataFrame({'col': [1, 2, 1]})
    df = label_categorical(df, ['col'])
    assert df['col'].dtype.name == 'category'

def test_engineer_datetime_features():
    df = pd.DataFrame({'order_dow': [0, 1, 6], 'order_hour_of_day': [5, 10, 15]})
    df = engineer_datetime_features(df)
    assert 'is_weekend' in df.columns
    assert 'is_morning_order' in df.columns
    assert list(df['is_weekend']) == [1, 0, 1]
    assert list(df['is_morning_order']) == [0, 1, 0]
