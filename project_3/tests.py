import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import unittest
import pandas as pd
from src.data_preprocessing import set_datatype

class TestDataPreprocessing(unittest.TestCase):
    def test_set_datatype_datetime(self):
        # Setup
        series = pd.Series(["2021-01-01", "2021-02-01"])
        
        # Test default conversion to datetime
        result_default = set_datatype(series)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result_default))
        
        # Test explicit datetime str
        result_explicit = set_datatype(series, "datetime64[ns]")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result_explicit))
        
    def test_set_datatype_int(self):
        # Setup
        series = pd.Series(["1", "2", "3"])
        
        # Test explicit int conversion
        result = set_datatype(series, "int")
        self.assertTrue(pd.api.types.is_integer_dtype(result))

if __name__ == '__main__':
    unittest.main()
