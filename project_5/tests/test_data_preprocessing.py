import pytest
import pandas as pd
from src.data_preprocessing import make_lowercase

def test_make_lowercase():
    # Setup
    df = pd.DataFrame({'Name': ['A', 'B'], 'Value_Int': [1, 2]})
    original_columns = df.columns.tolist()
    
    # Action
    make_lowercase(df)
    
    # Assert
    assert df.columns.tolist() == ['name', 'value_int']
    assert len(df.columns) == len(original_columns)
