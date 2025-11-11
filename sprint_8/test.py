''' Pick the best model for predicting binary classifier with a significant minority ratio, under various compensation strategies. 
compensation strategies: 'balanced weights' logistic regression setting, upsampling, downsampling'''

# libraries
from src_v_2.Input_Output.IO import Input
from src.data_explorers import view, see
from src_v_2.Input_Output.Cleaner import Cleaner
from src_v_2.Modeling.Model import Model

# load data
path = 'data/Churn.csv'
df = Input.from_csv(file_path=path)


## EDA
view(df)

# 'I'll keep the header names, 
# encode categorical, 
# ['Exit'] has minority of 20%, which I think is fine. especially out of 10k rows. 

# columns=['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
# dropping irrelevant rows and duplicates - if any. 
#df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)
Cleaner = Cleaner(df)

Cleaner.drop(['RowNumber', 'CustomerId', 'Surname'])
#df = df.drop_duplicates()
Cleaner.drop_duplicates()
df = Cleaner.df

# visualizer
#see(df)

## data transformation:
# encode categorical
# Note: ['Exit'] has minority of 20% and will stay that way:

#define target & identify ordinal categorical vars
# target = df['Exited']
n_rows = 10
print(n_rows)
Cleaner.set_rows(n_rows = n_rows)
Cleaner.set_missing(fill_method = 'drop', fill_value = None)

cleaned_df = Cleaner.df

