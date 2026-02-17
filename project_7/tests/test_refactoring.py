import pandas as pd
import unittest
import os
import shutil
from src.data.loader import load_data
from src.data.explorer import DataExplorer
from src.models.trainer import ModelTrainer, split_data
from sklearn.linear_model import LogisticRegression

class TestRefactoring(unittest.TestCase):
    def setUp(self):
        # Create a dummy csv
        self.test_csv = 'test_data.csv'
        pd.DataFrame({'a': [1,2,3,4,5], 'b': [0,1,0,1,0]}).to_csv(self.test_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

    def test_load_data(self):
        df = load_data(self.test_csv)
        self.assertEqual(len(df), 5)

    def test_explorer_structure(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        explorer = DataExplorer(df)
        summary = explorer.get_summary()
        self.assertIn('a', summary['Column'].values)
        self.assertIn('b', summary['Column'].values)

    def test_model_training(self):
        X = pd.DataFrame({'f1': [1, 2, 3, 4], 'f2': [2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1], name='target')
        trainer = ModelTrainer(model=LogisticRegression())
        trainer.train(X, y)
        scan = trainer.evaluate(X, y)
        self.assertIsInstance(scan, float)
    
    def test_split_data(self):
        df = pd.DataFrame({'f1': range(100), 'target': [0, 1] * 50})
        X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(df, 'target', test_size=0.4)
        
        # Train should be 60
        self.assertEqual(len(X_train), 60)
        # Valid should be 20 (half of 40)
        self.assertEqual(len(X_valid), 20)
        # Test should be 20
        self.assertEqual(len(X_test), 20)

if __name__ == '__main__':
    unittest.main()
