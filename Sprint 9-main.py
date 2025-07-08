import pandas as pd
from data_explorers import view


df_1 = pd.read_csv(r'C:\Users\Angelo\Documents\vscode\portfolio\portfolio-1\sprint 9 geo_data_0.csv')
df_2 = pd.read_csv(r'C:\Users\Angelo\Documents\vscode\portfolio\portfolio-1\sprint 9 geo_data_0.csv')
df_3 = pd.read_csv(r'C:\Users\Angelo\Documents\vscode\portfolio\portfolio-1\sprint 9 geo_data_0.csv')

view([df_1, df_2, df_3], view='headers')