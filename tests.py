import pandas as pd
from data_explorers import view, see
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
region_stats = pd.DataFrame(columns=['mean predicted reserves', #
                                    'RMSE', # 
                                    'Model score', # 
                                    'Mean profit', # 
                                    'Standard deviation of profit', # 
                                    '95% confidence interval for profit', # 
                                    'Samples mean', #
                                    'Samples std', # 
                                    '95% confidence interval for samples profit', #
                                    'risk of loss'], #
                                    index=['region_1', 'region_2', 'region_3'])
region_profit_samples = {
    'region_1': [],
    'region_2': [],
    'region_3': []
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
    top_200_wells = sorted_predictions.head(200)

    #  2.4. Print the average volume of predicted reserves and model RMSE.
    # Average volume of predicted reserves.
    average_volume = np.mean(predictions)
    region_stats.loc[region, 'mean predicted reserves'] = average_volume
    print(f'Average volume of predicted reserves for {region}: {average_volume:.2f}')

    # Compute the mean squared error between actual and predicted values.
    mse = mean_squared_error(validation[target], predictions)
    region_stats.loc[region, 'RMSE'] = np.sqrt(mse)

    # RMSE is the square root of the MSE.
    rmse = np.sqrt(mse)
    region_stats.loc[region, 'RMSE'] = rmse
    print(f'Model RMSE for {region}: {region_stats.loc[region, "RMSE"]:.2f}')

    #  2.5. Analyze the results.
    score = model.score(validation[features], validation[target])
    
    
    
    region_stats.loc[region, 'Model score'] = score
    print(f'Model score for {region}: {region_stats.loc[region, "Model score"]:.2f}')
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

    #  4.1. Pick the wells with the highest values of predictions. 
    

    #  4.2. Summarize the target volume of reserves in accordance with these predictions
    region_stats.loc[region, 'Mean profit'] = np.mean(validation['predictions'])
    print(f'Mean profit for {region}: {region_stats.loc[region, "Mean profit"]}')
    region_stats.loc[region, 'Standard deviation of profit'] = np.std(validation['predictions'])
    print(f'Standard deviation of profit for {region}: {region_stats.loc[region, "Standard deviation of profit"]}')
    region_stats.at[region, '95% confidence interval for profit'] = np.percentile(validation['predictions'], [2.5, 97.5]).tolist()
    print(f'95% confidence interval for profit for {region}: {region_stats.at[region, "95% confidence interval for profit"]}')
    #  4.3. Provide findings: suggest a region for oil wells' development and justify the choice. Calculate the profit for the obtained volume of reserves.
    # Calculate risks and profit for each region:
    #      5.1. Use the bootstrapping technique with 1000 samples to find the distribution of profit.
    
    for sample in range(1000):
        #bootstrap the validation set to find the distribution of profit
        bootstrap_sample = validation.sample(n=len(validation), replace=True)
        bootstrap_predictions = model.predict(bootstrap_sample[features])
        bootstrap_sample['predictions'] = bootstrap_predictions
        region_profit_samples[region].append(profit(bootstrap_sample['predictions'], wells_to_select=WELLS_TO_SELECT))
        
        

    top_200_total_profit = region_profit_samples[region]

    #      5.2. Find average profit, 95% confidence interval and risk of losses. Loss is negative profit, calculate it as a probability and then express as a percentage.
    region_stats.loc[region, 'Samples mean'] = np.mean(top_200_total_profit)
    region_stats.loc[region, 'Samples std'] = np.std(top_200_total_profit)
    region_stats.at[region, '95% confidence interval for samples profit'] = np.percentile(top_200_total_profit, [2.5, 97.5]).tolist()
    ci_lower = region_stats.loc[region, '95% confidence interval for samples profit'][0]
    ci_upper = region_stats.loc[region, '95% confidence interval for samples profit'][1]
    region_stats.loc[region, 'risk of loss'] = np.count_nonzero(np.array(top_200_total_profit) < 0) / len(top_200_total_profit) * 100
    
    print(f'samples mean: {region_stats.loc[region, "Samples mean"]}')
    print(f'samples std: {region_stats.loc[region, "Samples std"]}')
    print(f'95% confidence interval for samples profit: {region_stats.loc[region, "95% confidence interval for samples profit"]}')
    print(f'risk of loss: {region_stats.loc[region, "risk of loss"]}')


    #      5.3. Provide findings: suggest a region for development of oil wells and justify the choice.
print(region_stats)
top_region = region_stats['Mean profit'].idxmax()
print(f'Region with the highest mean profit: {region_stats.loc[top_region]}')