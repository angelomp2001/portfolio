import pandas as pd

output = pd.read_csv('output.csv', header=None)
output.columns = ['Test', 'normalize', 'lemmatize', 'stopword', 
                  'tokenizer', 'model', 'rows', 'F1', 
                  'ROC AUC', 'APS', 'Accuracy']
print(output.sort_values(by=['Accuracy', 'F1', 'ROC AUC', 'APS'], ascending=False).head(10))

'''
    Test  normalize  lemmatize  stopword tokenizer                   model  rows    F1  ROC AUC   APS  Accuracy
24     5      False      False     False      BERT      LogisticRegression  3000  1.00     1.00  1.00      1.00
28     9      False      False     False      BERT          LGBMClassifier  3000  0.86     0.94  0.95      0.88
33    14       True      False     False      BERT  RandomForestClassifier  3000  0.86     0.88  0.92      0.88
5      5      False      False     False      BERT      LogisticRegression  3000  0.84     0.93  0.93      0.85
1      1      False      False     False    tf_idf      LogisticRegression  3000  0.83     0.93  0.92      0.84
2      2       True      False     False    tf_idf      LogisticRegression  3000  0.83     0.93  0.92      0.84
3      3       True       True     False    tf_idf      LogisticRegression  3000  0.83     0.92  0.92      0.84
6      6       True      False     False      BERT      LogisticRegression  3000  0.82     0.91  0.91      0.83
4      4       True       True      True    tf_idf      LogisticRegression  3000  0.81     0.92  0.92      0.83
7      7       True       True     False      BERT      LogisticRegression  3000  0.81     0.90  0.90      0.82
'''