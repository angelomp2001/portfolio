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
compare_name_total_sales_x_platform = (
    name_multi_platform_total_sales.pivot(
        index = 'name',
        columns = 'platform',
        values = 'total_sales'
    )
)
compare_name_total_sales_x_platform = compare_name_total_sales_x_platform.fillna(value = 0)
print(compare_name_total_sales_x_platform.head())

# genre performance
genre = df_relevant.groupby('genre')['total_sales'].sum().sort_values(ascending = False)

# Top 5 genres by total sales
print(genre.head())

# genre distribution
plt.figure(figsize=(50, 10))
genre.plot(
    kind = 'bar',
    ylabel = 'total_sales'
)
plt.xticks(
    rotation = 0,
    fontsize=30)
plt.yticks(fontsize = 30)
plt.ylabel('total_sales',fontsize = 30)
plt.show()

# market share for each genre
market_size = genre.sum()
genre_df = genre.to_frame()
genre_df['percent'] = genre_df['total_sales']/market_size
for genre_name, percent in zip(genre_df.index, genre_df['percent']):
  print(genre_name, round(percent,2)*100,"%")

# performance of the genre across time.
year_genre = df_relevant.groupby(['genre', 'year_of_release'])['total_sales'].agg(['sum']) #.agg() keeps it as df.
year_genre.reset_index(inplace = True)
year_genre = year_genre.rename(columns={'year_of_release': 'year','sum':'total_sales'})
year_genre['year'] = year_genre['year'].dt.strftime('%Y')
year_genre = year_genre.pivot(
    index='year',
    columns = 'genre',
    values = 'total_sales'
)

#average genre sales per year (row)
year_genre['average_genre_sales_year'] = year_genre.mean(axis=1) #mean from column values

#total sales of each genre across the years
year_genre.loc['total_sales'] = year_genre.drop('average_genre_sales_year', axis=1).sum() #add row 'total Sales' by genre - excludes 'average_genre_sales_year' in calculation

#average sales
year_genre.loc['average_sales'] = year_genre.drop('total_sales', axis=0).mean(axis=0)

# a year from two months ago
see(year_genre[-12:-2])

# Function to analyze platform performance by region
def performance(platform, region):
  #groupby row for col agg
  agg_dict = {
      'name' : 'count',
      'platform' : 'count',
      'year_of_release': 'count',
      'genre' : 'count',
      'na_sales':'sum',
      'eu_sales':'sum',
      'jp_sales':'sum',
      'other_sales':'sum',
      'critic_score':'sum',
      'user_score':'sum',
      'rating' : 'count',
      'total_sales':'sum'
  }

  performance = df_relevant.groupby(['platform'])[region].agg(agg_dict[region])
  print(f'{platform}:',performance.loc[platform])

# performance of PS3
performance('PS3','other_sales')

# Analyze each region
performance('PS3','na_sales')
performance('PS3','eu_sales')
performance('PS3','jp_sales')
performance('PS3','other_sales')

# comparative platform analysis
agg_dict = {
#    'name' : 'count',
#    'platform' : 'count',
#    'year_of_release': 'count',
#    'genre' : 'count',
    'na_sales':'sum',
    'eu_sales':'sum',
    'jp_sales':'sum',
    'other_sales':'sum',
#    'critic_score':'sum',
#    'user_score':'sum',
#    'rating' : 'count',
#    'total_sales':'sum'
}

platform_sales = df_relevant.groupby(['platform'])[['na_sales','eu_sales','jp_sales','other_sales']].agg(agg_dict)
print(platform_sales = df_relevant.groupby(['platform'])[['na_sales','eu_sales','jp_sales','other_sales']].agg(agg_dict)
)

# Visualize cross-regional comparison for top platforms
# Create the heatmap
plt.figure(figsize=(15, 10))
plt.imshow(platform_sales, cmap='YlGnBu', interpolation='nearest')

# color bar
plt.colorbar()

# Set labels and title
plt.xlabel('Platform')
plt.ylabel(performance.index.name)
plt.title('Heatmap of Platform Sales by Region')

plt.xticks(
    ticks=np.arange(len(performance.columns)),
    labels=performance.columns,
    rotation=45,
    ha='right'
)
plt.yticks(
    ticks=np.arange(len(performance.index)),
    labels=performance.index
)

# Show the plot
plt.show()

# nalyze genre performance by region
agg_dict = {
#    'name' : 'count',
#    'platform' : 'count',
#    'year_of_release': 'count',
#    'genre' : 'count',
    'na_sales':'sum',
    'eu_sales':'sum',
    'jp_sales':'sum',
    'other_sales':'sum',
#    'critic_score':'sum',
#    'user_score':'sum',
#    'rating' : 'count',
#    'total_sales':'sum'
}

genre_performance = df_relevant.groupby(['genre'])[['na_sales','eu_sales','jp_sales','other_sales']].agg(agg_dict)
print(genre_performance)

# heatmap
plt.figure(figsize=(15, 10))
plt.imshow(genre_performance, cmap='YlGnBu', interpolation='nearest')

#  color bar
plt.colorbar()

# Set labels and title
plt.xlabel('Platform')
plt.ylabel(genre_performance.index.name)
plt.title('Heatmap of Platform Sales by Region')

plt.xticks(
    ticks=np.arange(len(genre_performance.columns)),
    labels=genre_performance.columns,
    rotation=45,
    ha='right'
)
plt.yticks(
    ticks=np.arange(len(genre_performance.index)),
    labels=genre_performance.index
)

# Show plot
plt.show()

# comparative genre analysis
print(genre_performance)

# ESRB rating impact
agg_dict = {
#    'name' : 'count',
#    'platform' : 'count',
#    'year_of_release': 'count',
#    'genre' : 'count',
    'na_sales':'sum',
    'eu_sales':'sum',
    'jp_sales':'sum',
    'other_sales':'sum',
#    'critic_score':'sum',
#    'user_score':'sum',
#    'rating' : 'count',
#    'total_sales':'sum'
}

rating_performance = df_relevant.groupby(['rating'])[['na_sales','eu_sales','jp_sales','other_sales']].agg(agg_dict)
print(rating_performance)

# Analyze ESRB impact for each region
print(rating_performance)

#hypothesis testing
# alpha = 0.05
#—Average user ratings of the Xbox One and PC platforms are the same.
xbox_one_query ="platform == 'XOne'"
pc_query =  "platform == 'PC'"
xbox_one_scores = df_relevant.query(xbox_one_query)['user_score']
pc_scores = df_relevant.query(pc_query)['user_score']

xbox_one_vs_pc_result = st.ttest_ind(xbox_one_scores, pc_scores, nan_policy='omit', equal_var=False)
print(f'xbox_one_vs_pc_result p-value: {xbox_one_vs_pc_result.pvalue}\n the mean scores are not the same - reject H0')


#Average user ratings for the Action and Sports genres are the same.
action_query = 'genre == "Action"'
sports_query = 'genre == "Sports"'
action_scores = df_relevant.query(action_query)['user_score']
sports_scores = df_relevant.query(sports_query)['user_score']

action_vs_sports_results = st.ttest_ind(action_scores, sports_scores, nan_policy='omit', equal_var=False)
print(f'action_vs_sports_results p-value: {action_vs_sports_results.pvalue}\n the mean scores are the same - accept H0')

# general conclusion
# peak game releases was 2008
# peak sales was 2008
# peak mean sales per game was 1989
# new platform creates new sales, all platforms decline over time
# sales cycle on a platform is about 10 years.
# Top platforms by sales: PS3, X360, Wii
# YoY growth is usually strongest in the first year and declines over time
# Sales distributions by platform are not always normally distributed
# critic score and user score are correlated
# The top selling genre is Action
# Sales across all genres are declining
# NA followed by the EU do the most sales
# Na AND EU perform similarly
# jp and other perform similarly
# Genre sales vary by region
# ratings vary by region sales
# ratings appear associated with sales (higher rations, higher sales)
# Average user ratings are different for xbox and PC
# average user rations are not different for action vs Sports genre
# we could do a much better analysis if we had price and units sold.
