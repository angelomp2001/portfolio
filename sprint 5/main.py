from src.data_preprocessing import *

path = 'data/games.csv'
df = load_data(path)

#config view df
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 500)        # Increase horizontal width
pd.set_option('display.max_colwidth', None) # Show full content of each column

# view df
view(df)


# make lowercase
make_lowercase(df)

# relabel missing
relabel_missing(df)

#adjust dtype
adjust_dtypes(df)

#remove duplicates
drop_duplicates(df)

# Display basic information about the dataset
df.info()

# Verify the changes
print(df.columns)

# Check current data types
view(df, 'dtypes')
df.info()

view(df,'values')

view(df, 'missing values')

# Analyze patterns in missing values
#MAR by col
missing_cols = ['name', 'year_of_release', 'genre', 'critic_score', 'user_score', 'rating']
analyze(missing_cols)

# Total sales across all regions
total_sales(df)

# DataFrame with game releases by year
game_releases_by_year_results = game_releases_by_year(df)

# average game releases by year
average_game_releases_by_year_results = average_game_releases_by_year(df)

# Visualization of the distribution of games across years
# Assuming game_releases_by_year is your DataFrame
see(game_releases_by_year_results['name'])
see(game_releases_by_year_results[['na_sales','eu_sales','jp_sales','other_sales','total_sales']])
see(average_game_releases_by_year_results[['mean_na_sales','mean_eu_sales','mean_jp_sales','mean_other_sales']])

# Summary statistics for each year
summarize_by_year(df)

# Calculate total sales by platform and year
sales_by_platform_and_year(df)

# relevant years (3):
df_relevant = relevant_period(df, start_year=2013, end_year=2016)

# Critic Scores
view(df_relevant['critic_score'],'missing values')

# User Scores
view(df_relevant['user_score'],'missing values')

# Calculate correlations
relationship(df_relevant[['critic_score','user_score']])

# games released on multiple platforms
last_3_years = last_3_years_df(df)

#the number of unique platforms for each name.
name_by_platform = n_unique_platforms(df_relevant)

# new series where platform unique > 1
multi_platform_games = name_by_platform[name_by_platform > 1] 

# lists names on multiple platforms
print(view(multi_platform_games,'summaries')) 

#multi_platform_games #series of names on multiple platforms
multi_platform_games_list = multi_platform_games.index.to_list()

# sales across platforms for these games
# Group by 'name' and 'platform' then sum up total_sales
name_platform_total_sales_reset_index = name_platform_total_sales(df) 

# Apply filter for multi-platform games (ensure multi_platform_games is defined)
name_multi_platform_total_sales = name_platform_total_sales_reset_index[
    name_platform_total_sales_reset_index['name'].isin(multi_platform_games.index)
]
print(name_multi_platform_total_sales.shape[0])

# compare_name_total_sales_by_platform
compare_name_total_sales_x_platform = compare_name_total_sales_x_platform(name_multi_platform_total_sales)

# genre performance
genre = genre_performance(df_relevant)


# market share for each genre
year_genre = market_share(genre)

# a year from two months ago
see(year_genre[-12:-2])

# performance of PS3
performance('PS3','other_sales', df_relevant)

# Analyze each region
performance('PS3','na_sales', df_relevant)
performance('PS3','eu_sales', df_relevant)
performance('PS3','jp_sales', df_relevant)
performance('PS3','other_sales', df_relevant)

# comparative platform analysis
platform_sales = comparative_platform_analysis(df_relevant)

# analyze genre performance by region
genre_performance_region = genre_performance_by_region(df_relevant)

# comparative genre analysis
print(genre_performance_region)

# ESRB rating impact
rationg_performance = esrb_impact(df_relevant)

#hypothesis testing
hypothesis_testing(df_relevant)

'''
general conclusion
peak game releases was 2008
peak sales was 2008
peak mean sales per game was 1989
new platform creates new sales, all platforms decline over time
sales cycle on a platform is about 10 years.
Top platforms by sales: PS3, X360, Wii
YoY growth is usually strongest in the first year and declines over time
Sales distributions by platform are not always normally distributed
critic score and user score are correlated
The top selling genre is Action
Sales across all genres are declining
NA followed by the EU do the most sales
Na AND EU perform similarly
jp and other perform similarly
Genre sales vary by region
ratings vary by region sales
ratings appear associated with sales (higher rations, higher sales)
Average user ratings are different for xbox and PC
average user rations are not different for action vs Sports genre
we could do a much better analysis if we had price and units sold.
'''