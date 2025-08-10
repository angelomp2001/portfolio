'''
Find the best place to look for oil, accounting for profit and risk of loss.
'''
#libraries
import pandas as pd
from data_explorers import view, see
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# load data
df_1 = pd.read_csv('sprint 9 geo_data_0.csv')
df_2 = pd.read_csv('sprint 9 geo_data_1.csv')
df_3 = pd.read_csv('sprint 9 geo_data_2.csv')


## EDA
view(df_1, view='headers')
view(df_2, view='headers')
view(df_3, view='headers')


# drop duplicates based on 'id' column
df_1 = df_1.drop_duplicates(subset = 'id')
df_2 = df_2.drop_duplicates(subset = 'id')
df_3 = df_3.drop_duplicates(subset = 'id')


# drop 'id' column
df_1 = df_1.drop(columns=['id'])
df_2 = df_2.drop(columns=['id'])
df_3 = df_3.drop(columns=['id'])

# group df
dfs = {
    'region_1': df_1,
    'region_2': df_2,
    'region_3': df_3
}

# visualize data
see(df_1)
see(df_2)
see(df_3)


## initialize tables
# statistics table
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

# table to store profit samples
region_profit_samples = {
    'region_1': [],
    'region_2': [],
    'region_3': []
}


#constants provided by the client:
BUDGET = 100_000_000  # $100 million
WELLS_TO_SELECT = 200  # Top 200 wells
REVENUE_PER_BARREL = 4.5  # $4.5 per barrel
REVENUE_PER_UNIT = 4500  # $4,500 per thousand barrels
POINTS_STUDIED = 500  # Study 500 points per region
model = LinearRegression()
random_state = 42

## functions:
# profit
def profit(
            target: pd.Series,
            predictions: pd.Series,
            wells_to_select: int = WELLS_TO_SELECT
            ) -> float:
        '''
            Calculate profit based on predictions and target values.
            Selects top wells based on predictions and calculates total profit.
        '''
        # select top predictions
        top_predictions = predictions.nlargest(wells_to_select)

        # map top predictions to target values in target
        target_of_predictions = target.loc[top_predictions.index]

        # calculate total revenue of top predictions
        total_revenue = target_of_predictions.to_numpy().sum() * REVENUE_PER_UNIT

        # calculate total profit
        total_profit = total_revenue - BUDGET

        # print total profit
        print(f"Total profit from selected wells: {total_profit:.2f}")
        return total_profit

# predict product
def product_predictions(
          training: pd.DataFrame,
          validation: pd.DataFrame,
          model: None = None
        ) -> pd.Series:
        '''
            Predict how much product each well will produce.
        '''
        # fit model
        model.fit(training[features], training[target])

        # predict product for validation set
        validation['predictions'] = model.predict(validation[features])     
        
        return validation

# calculate statistics
def stats(
            actual: pd.DataFrame,
            predictions: pd.Series,
            wells_to_select=WELLS_TO_SELECT) -> pd.Series:
    '''
        Calculate statistics for the predictions.
    '''
    # calculate mean predicted reserves
    average_volume = np.mean(predictions)

    # store mean predicted reserves in region_stats
    region_stats.loc[region, 'mean predicted reserves'] = average_volume
    print(f'Average volume of predicted reserves for {region}: {average_volume:.2f}')

    # calculate and store RMSE
    mse = mean_squared_error(validation[target], predictions)
    region_stats.loc[region, 'RMSE'] = np.sqrt(mse)
    
    # select top 200 wells based on predictions
    top_200_wells = predictions.nlargest(wells_to_select)

    ## calculate total revenue and profit and stats
    total_revenue = top_200_wells.sum() * REVENUE_PER_UNIT
    total_profit = total_revenue - BUDGET
    mean_profit = np.mean(total_profit)
    std_profit = np.std(total_profit)
    ci = np.percentile(total_profit, [2.5, 97.5])
    rmse = np.sqrt(mse)
    region_stats.loc[region, 'RMSE'] = rmse
    
    # store model score
    score = model.score(validation[features], validation[target])
    region_stats.loc[region, 'Model score'] = score
    print(f'Model score for {region}: {region_stats.loc[region, "Model score"]:.2f}')

    # print RMSE
    print(f'Model RMSE for {region}: {region_stats.loc[region, "RMSE"]:.2f}')
    MIN_VOL_PER_WELL = BUDGET / (REVENUE_PER_UNIT * WELLS_TO_SELECT)
    print(f"Minimum volume per well: {MIN_VOL_PER_WELL:.2f}")
    
    # print profit
    region_stats.loc[region, 'Mean profit'] = profit(target=actual, predictions=predictions, wells_to_select=WELLS_TO_SELECT)
    print(f'Mean profit for {region}: {region_stats.loc[region, "Mean profit"]}')
    
    # store standard deviation of profit and confidence interval
    region_stats.loc[region, 'Standard deviation of profit'] = np.std(validation['predictions'])
    print(f'Standard deviation of profit for {region}: {region_stats.loc[region, "Standard deviation of profit"]}')
    
    # store 95% confidence interval for profit
    region_stats.at[region, '95% confidence interval for profit'] = np.percentile(validation['predictions'], [2.5, 97.5]).tolist()
    print(f'95% confidence interval for profit for {region}: {region_stats.at[region, "95% confidence interval for profit"]}')
    
    return pd.Series({
        'mean profit': mean_profit,
        'std profit': std_profit,
        '95% confidence interval': ci
    })

def bootstrap_predictions(
    dataframe: pd.DataFrame,
    repeats: int = 1000,
    random_state: int = 42
    ):
    # Create a random number from a number
    rng = np.random.RandomState(random_state)
    
    # Example: initialize a dictionary to store profit samples for a given region
    region_profit_samples = {"region1": []}
    region = "region1"

    # Perform bootstrapping
    for sample in range(repeats):
        # create a random number per sample
        sample_seed = rng.randint(0, 10000)
        # create a bootstrap sample of predictions from the validation set
        bootstrap_sample = dataframe.sample(n=POINTS_STUDIED , replace=True, random_state=sample_seed)
        region_profit_samples[region].append(profit(target=bootstrap_sample['product'], predictions=bootstrap_sample['predictions'], wells_to_select=WELLS_TO_SELECT))
        
    return region_profit_samples[region]

def bootstrap_stats(
          top_200_total_profit: list,
          region: str,
          region_stats: pd.DataFrame
          ) -> None:

        # calculate and store bootstrap statistics 
        region_stats.loc[region, 'Samples mean'] = np.mean(top_200_total_profit)
        region_stats.loc[region, 'Samples std'] = np.std(top_200_total_profit)
        region_stats.at[region, '95% confidence interval for samples profit'] = np.percentile(top_200_total_profit, [2.5, 97.5]).tolist()
        region_stats.loc[region, 'risk of loss'] = np.count_nonzero(np.array(top_200_total_profit) < 0) / len(top_200_total_profit) * 100
    
        print(f'samples mean: {region_stats.loc[region, "Samples mean"]}')
        print(f'samples std: {region_stats.loc[region, "Samples std"]}')
        print(f'95% confidence interval for samples profit: {region_stats.loc[region, "95% confidence interval for samples profit"]}')
        print(f'risk of loss: {region_stats.loc[region, "risk of loss"]}')


# for each df
for region, df in dfs.items():
    print(f'Region: {region}')
    training, validation = train_test_split(df, test_size=0.25, random_state=42)
    
    # defining features and target
    target = training['product'].name
    features = training.drop(columns=target).columns

    # calculate predictions
    validation_target_predictions  = product_predictions(training=training, validation=validation, model=model)

    # calculate stats
    stats(actual=validation_target_predictions['product'], predictions=validation_target_predictions['predictions'], wells_to_select=WELLS_TO_SELECT)

    # calculate profit on predictions
    top_200_total_profit = bootstrap_predictions(dataframe=validation_target_predictions, repeats=1000, random_state=random_state)
    plt.hist(top_200_total_profit, bins=50, alpha=0.75)
    plt.show()

    # calculate bootstrap stats
    bootstrap_stats(top_200_total_profit, region, region_stats)


# Findings
print(region_stats)
region_stats['Mean profit'] = pd.to_numeric(region_stats['Mean profit'], errors='coerce')
top_region = region_stats['Mean profit'].idxmax()
print(f'Region with the highest mean profit: {region_stats.loc[top_region]}')

'''
Conclusion
We analyzed all three regions.

Region 1 contains the most amount of product and profit, on average.

Its mean profit is $34,685,297.87, and its risk of loss is 1.5%.

Model prediction quality for region 1 was low at 27%.

Bootstrapping was used for all regions to capture variability around region profitability.

We calculated profit using the top 200 most profitable wells from repeated sampling of 500 wells to simulate realistic business scenarios.

When accounting for profit variability, region 2 offered the highest profit at $6,450,501.83.

Furthermore, region 2 offered the lowest risk of loss at 0.6% (95% of profit values ranged from 
11,879,197.54).

In conclusion, region 2 is proposed to minimize risk of loss.
'''