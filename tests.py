from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

df_1 = pd.read_csv(r'C:\Users\Angelo\Documents\vscode\portfolio\portfolio-1\sprint 9 geo_data_0.csv')
df_1 = df_1.drop(columns=['id'])
# df_2 = df_2.drop(columns=['id'])
# df_3 = df_3.drop(columns=['id'])



# goal: max(oil wells value)
# value: quality and volume
# target: volume in new wells

# region: max(total profit for the selected oil wells)
# df = 1 region of samples

target = df_1['product'].name
features = df_1.drop(columns=target).columns

df_1_training, df_1_validation = train_test_split(df_1, test_size=0.25, random_state=42)
df_1_training_target = df_1_training[target]
df_1_training_features = df_1_training[features]

model = LinearRegression()
model.fit(df_1_training_features, df_1_training_target)

scores = model.score(df_1_validation[features], df_1_validation[target])
print(f"Model score: {scores:.2f}")

# Predict on the validation set using the correct features.
predictions = model.predict(df_1_validation[features])

# Compute the mean squared error between actual and predicted values.
mse = mean_squared_error(df_1_validation[target], predictions)

# RMSE is the square root of the MSE.
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")