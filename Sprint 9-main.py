import pandas as pd
from data_explorers import view, see
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error

df_1 = pd.read_csv(r'C:\Users\Angelo\Documents\vscode\portfolio\portfolio-1\sprint 9 geo_data_0.csv')
df_2 = pd.read_csv(r'C:\Users\Angelo\Documents\vscode\portfolio\portfolio-1\sprint 9 geo_data_0.csv')
df_3 = pd.read_csv(r'C:\Users\Angelo\Documents\vscode\portfolio\portfolio-1\sprint 9 geo_data_0.csv')

# Project instructions
# Download and prepare the data. Explain the procedure.
# Train and test the model for each region:
#  2.1. Split the data into a training set and validation set at a ratio of 75:25.
#  2.2. Train the model and make predictions for the validation set.
#  2.3. Save the predictions and correct answers for the validation set.
#  2.4. Print the average volume of predicted reserves and model RMSE.
#  2.5. Analyze the results.
# Prepare for profit calculation:
#  3.1. Store all key values for calculations in separate variables.
#  3.2. Calculate the volume of reserves sufficient for developing a new well without losses. Compare the obtained value with the average volume of reserves in each region.
#  3.3. Provide the findings about the preparation for profit calculation step.
# Write a function to calculate profit from a set of selected oil wells and model predictions:
#  4.1. Pick the wells with the highest values of predictions. 
#  4.2. Summarize the target volume of reserves in accordance with these predictions
#  4.3. Provide findings: suggest a region for oil wells' development and justify the choice. Calculate the profit for the obtained volume of reserves.
# Calculate risks and profit for each region:
#      5.1. Use the bootstrapping technique with 1000 samples to find the distribution of profit.
#      5.2. Find average profit, 95% confidence interval and risk of losses. Loss is negative profit, calculate it as a probability and then express as a percentage.
#      5.3. Provide findings: suggest a region for development of oil wells and justify the choice.

# goal: max(oil wells value)
# value: quality and volume
# target: volume in new wells

view(df_1, view='headers')
view(df_2, view='headers')
view(df_3, view='headers')

# we drop ['id'] as we don't need it for anything

df_1 = df_1.drop(columns=['id'])
df_2 = df_2.drop(columns=['id'])
df_3 = df_3.drop(columns=['id'])
dfs = {
    'region_1': df_1,
    'region_2': df_2,
    'region_3': df_3
}

# see(df_1)
# see(df_2)
# see(df_3)

# don't assume target is normal. 

# Cross-validation and bootstrapping are not needed here as we have lots of data.  

#2.1 Split the data into a training set and validation set at a ratio of 75:25.
region_stats = {
    'region_1': {'mean predicted reserves': 0, 'RMSE': 0, 'model score': 0, 'Mean profit': 0, 'Standard deviation of profit': 0, '95% confidence interval for profit': (0, 0), 'risk of loss': 0},
    'region_2': {'mean predicted reserves': 0, 'RMSE': 0, 'model score': 0, 'Mean profit': 0, 'Standard deviation of profit': 0, '95% confidence interval for profit': (0, 0), 'risk of loss': 0},
    'region_3': {'mean predicted reserves': 0, 'RMSE': 0, 'model score': 0, 'Mean profit': 0, 'Standard deviation of profit': 0, '95% confidence interval for profit': (0, 0), 'risk of loss': 0}
}
for region, df in dfs.items():
    print(f'Region: {region}')
    training, validation = train_test_split(df_1, test_size=0.25, random_state=42)

    #  2.2. Train the model and make predictions for the validation set.
    # define features and target
    target = df['product'].name
    features = df.drop(columns=target).columns

    # define features and target for training and validation sets
    training_target = training[target]
    training_features = training[features]

    # Train the model using Linear Regression.
    model = LinearRegression()
    model.fit(training_features, training_target)

    #  2.3. Save the predictions and correct answers for the validation set.
    # Predict on the validation set using the correct features.
    predictions = model.predict(validation[features])
    validation['predictions'] = predictions
    sorted_predictions = validation['predictions'].sort_values(ascending=False)

    
    # Note: In a real-world scenario, you would want to save the predictions to a file or database for further analysis.
    top_200_wells = sorted_predictions.head(200)

    #  2.4. Print the average volume of predicted reserves and model RMSE.
    # Average volume of predicted reserves.
    average_volume = np.mean(predictions)
    print(f"Average volume of predicted reserves: {average_volume:.2f}")     

    # Compute the mean squared error between actual and predicted values.
    mse = mean_squared_error(validation[target], predictions)

    # RMSE is the square root of the MSE.
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.2f}")

    #  2.5. Analyze the results.
    scores = model.score(validation[features], validation[target])
    print(f"Model score: {scores:.2f}")
    # this model is horrible.  We should bootstrap the predictions to get a better estimate of the model's performance.

    # Prepare for profit calculation:
    #  3.1. Store all key values for calculations in separate variables.
    BUDGET = 100_000_000  # $100 million
    WELLS_TO_SELECT = 200  # Top 200 wells
    REVENUE_PER_BARREL = 4.5  # $4.5 per barrel
    REVENUE_PER_UNIT = 4500  # $4,500 per thousand barrels
    POINTS_STUDIED = 500  # Study 500 points per region

    #  3.2. Calculate the volume of reserves sufficient for developing a new well without losses. Compare the obtained value with the average volume of reserves in each region.
    MIN_VOL_PER_WELL = BUDGET / (REVENUE_PER_UNIT * WELLS_TO_SELECT)
    #print(f"Minimum volume per well: {MIN_VOL_PER_WELL:.2f}")

    #  3.3. Provide the findings about the preparation for profit calculation step.
    # Write a function to calculate profit from a set of selected oil wells and model predictions:
    def profit(predictions, wells_to_select=WELLS_TO_SELECT):
        top_200_wells = predictions.nlargest(wells_to_select)
        #print(f"Top {top_200_wells} wells selected based on predictions.")
        total_revenue = top_200_wells.sum() * REVENUE_PER_UNIT
        total_profit = total_revenue - BUDGET
        #print(f"Total profit from selected wells: {total_profit:.2f}")
        return total_profit

    profit(validation['predictions'], wells_to_select=WELLS_TO_SELECT)


    #  4.1. Pick the wells with the highest values of predictions. 
    top_200_total_profit = []
    for sample in range(1000):
        #bootstrap the validation set to find the distribution of profit
        bootstrap_sample = validation.sample(n=len(validation), replace=True)
        bootstrap_predictions = model.predict(bootstrap_sample[features])
        bootstrap_sample['predictions'] = bootstrap_predictions
        top_200_total_profit.append(profit(bootstrap_sample['predictions'], wells_to_select=WELLS_TO_SELECT))
        

    top_200_total_profit = np.array(top_200_total_profit)

    #  4.2. Summarize the target volume of reserves in accordance with these predictions
    mean_profit = np.mean(top_200_total_profit)
    std_profit = np.std(top_200_total_profit)
    ci_lower, ci_upper = np.percentile(top_200_total_profit, [2.5, 97.5])

    print(f"Mean profit: {mean_profit:.2f}")
    print(f"Standard deviation of profit: {std_profit:.2f}")
    print(f"95% confidence interval for profit: ({ci_lower:.2f}, {ci_upper:.2f})")
    
    
    #  4.3. Provide findings: suggest a region for oil wells' development and justify the choice. Calculate the profit for the obtained volume of reserves.
    # Calculate risks and profit for each region:
    #      5.1. Use the bootstrapping technique with 1000 samples to find the distribution of profit.
    #      5.2. Find average profit, 95% confidence interval and risk of losses. Loss is negative profit, calculate it as a probability and then express as a percentage.
    #      5.3. Provide findings: suggest a region for development of oil wells and justify the choice.