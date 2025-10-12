'''
Find the best place to look for oil, accounting for profit and risk of loss.
'''
from src.data_explorers import *
from src.data_preprocessing import *

# load data
geo_data_0 = 'data/geo_data_0.csv'
geo_data_1 = 'data/geo_data_1.csv'
geo_data_2 = 'data/geo_data_2.csv'

df_1 = load_data(geo_data_0)
df_2 = load_data(geo_data_1)
df_3 = load_data(geo_data_2)

## EDA
view(df_1, view='headers')
view(df_2, view='headers')
view(df_3, view='headers')


#preprocess data
df_1, df_2, df_3, dfs = preprocess_data(df_1, df_2, df_3)

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
BUDGET, WELLS_TO_SELECT, REVENUE_PER_BARREL, REVENUE_PER_UNIT, POINTS_STUDIED, model, random_state = inputs(
    BUDGET = 100_000_000,  # $100 million
    WELLS_TO_SELECT = 200,  # Top 200 wells
    REVENUE_PER_BARREL = 4.5,  # $4.5 per barrel
    REVENUE_PER_UNIT = 4500,  # $4,500 per thousand barrels
    POINTS_STUDIED = 500,  # Study 500 points per region
    model = LinearRegression(),
    random_state = 42
)

# top 200 wells
top_200_wells(dfs, region_stats, model, random_state)

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