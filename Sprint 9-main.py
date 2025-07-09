import pandas as pd
from data_explorers import view, see
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df_1 = pd.read_csv(r'C:\Users\Angelo\Documents\vscode\portfolio\portfolio-1\sprint 9 geo_data_0.csv')
df_2 = pd.read_csv(r'C:\Users\Angelo\Documents\vscode\portfolio\portfolio-1\sprint 9 geo_data_0.csv')
df_3 = pd.read_csv(r'C:\Users\Angelo\Documents\vscode\portfolio\portfolio-1\sprint 9 geo_data_0.csv')

view(df_1, view='headers')
view(df_2, view='headers')
view(df_3, view='headers')

# we drop id as we don't need it for anything

df_1 = df_1.drop(columns=['id'])
df_2 = df_2.drop(columns=['id'])
df_3 = df_3.drop(columns=['id'])


see(df_1)
# see(df_2)
# see(df_3)

# goal: max(oil wells value)
# value: quality and volume
# target: volume in new wells

# region: max(total profit for the selected oil wells)
# df = 1 region of samples

df_1_training, df_1_validation = train_test_split(df_1, test_size=0.25, random_state=42)
df_2_training, df_2_validation = train_test_split(df_2, test_size=0.25, random_state=42)
df_3_training, df_3_validation = train_test_split(df_3, test_size=0.25, random_state=42)