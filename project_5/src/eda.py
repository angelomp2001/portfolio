import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st



def analyze(df, missing_cols):
    for missing in missing_cols:
        print(f'{missing}: check for constant values in other rows:\n',
            df[df[missing_cols[missing_cols.index(missing)]].isna()]
            )
    
    #skipping name because it's only 2 values and I can't guess what they are.
    #Values found in name, although I noticed row 16711 does not follow this pattern
    print(df.iloc[183]['year_of_release'])
    df.at[183,'year_of_release'] = pd.to_datetime('2004-01-01')
    print(df.iloc[183]['year_of_release'])
    df.at[377,'year_of_release'] = pd.to_datetime('2004-01-01')
    df.at[475,'year_of_release'] = pd.to_datetime('2006-01-01')
    df.at[16373  ,'year_of_release'] = pd.to_datetime('2008-01-01')

    # genre skip
    # critic_score, no pattern visible, other than other scores/rating missing
    # user_score, no pattern visible, other than other scores/rating missing
    # rating, no pattern visible, other than other scores/rating missing

def total_sales(df):
    df['total_sales'] = df['na_sales'] + df['eu_sales'] + df['jp_sales'] + df['other_sales']
    df['total_sales'].describe()
    df['total_sales'].sum()

def game_releases_by_year(df):
    agg_dict = {
    'name': 'count',
    'platform': 'count',
    'genre': 'count',
    'na_sales': 'sum',
    'eu_sales': 'sum',
    'jp_sales': 'sum',
    'other_sales': 'sum',
    'critic_score': 'mean',
    'user_score': 'mean',
    'rating': 'count',
    'total_sales': 'sum'
    }
    game_releases_by_year = df.groupby('year_of_release').agg(agg_dict)
    print(game_releases_by_year.head(5))

    return game_releases_by_year

def average_game_releases_by_year(df):
    agg_dict = {
    'name': 'count',
    'platform': 'count',
    'genre': 'count',
    'na_sales': 'mean',
    'eu_sales': 'mean',
    'jp_sales': 'mean',
    'other_sales': 'mean',
    'critic_score': 'mean',
    'user_score': 'mean',
    'rating': 'count',
    'total_sales': 'mean'
}
    mean_game_releases_by_year = df.groupby('year_of_release').agg(agg_dict)
    mean_game_releases_by_year.rename(columns={
        'na_sales': 'mean_na_sales',
        'eu_sales': 'mean_eu_sales',
        'jp_sales': 'mean_jp_sales',
        'other_sales': 'mean_other_sales',
        'total_sales': 'mean_total_sales'
    }, inplace = True)
    print(mean_game_releases_by_year.head(5))

    return mean_game_releases_by_year

def see(col, x=None):
    # Determine the x-axis label based on the provided argument or index name
    if x is None:
        x_label = col.index.name or "Index"
    else:
        x_label = x

    # For a DataFrame, let pandas use the index without forcing an x column
    if isinstance(col, pd.DataFrame):
        ax = col.plot(
            kind='line',
            figsize=(12, 6),
            title=f"{', '.join(col.columns)} by {x_label}",
            grid=True
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Value")
    else:
        # For a Series, plot normally
        ax = col.plot(
            kind='line',
            figsize=(12, 6),
            title=f'{col.name} by {x_label}',
            grid=True
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(col.name)

    plt.show()

def summarize_by_year(df):
    #summary assuming normal
    agg_dict = {
        'name': 'count',
        'platform': 'count',
        'genre': 'count',
        'na_sales': ['mean', 'std'],
        'eu_sales': ['mean', 'std'],
        'jp_sales': ['mean', 'std'],
        'other_sales': ['mean', 'std'],
        'critic_score': ['mean', 'std'],
        'user_score': ['mean', 'std'],
        'rating': 'count',
        'total_sales': ['mean', 'std']
    }
    game_releases_by_year_normal = df.groupby('year_of_release').agg(agg_dict)

    #summary assuming skewed

    def quantile_25(x):
        return x.quantile(0.25)

    def quantile_75(x):
        return x.quantile(0.75)


    agg_dict = {
        'name': 'count',
        'platform': 'count',
        'genre': 'count',
        'na_sales': ['median', ('q1', quantile_25), ('q3', quantile_75)],
        'eu_sales': ['median', ('q1', quantile_25), ('q3', quantile_75)],
        'jp_sales': ['median', ('q1', quantile_25), ('q3', quantile_75)],
        'other_sales': ['median', ('q1', quantile_25), ('q3', quantile_75)],
        'critic_score': ['median', ('q1', quantile_25), ('q3', quantile_75)],
        'user_score': ['median', ('q1', quantile_25), ('q3', quantile_75)],
        'rating': 'count',
        'total_sales': ['median', ('q1', quantile_25), ('q3', quantile_75)]
    }
    game_releases_by_year_skewed = df.groupby('year_of_release').agg(agg_dict)

    # Left: Line plot
    #years = game_releases_by_year_normal.index
    mean_sales = game_releases_by_year_normal[('total_sales', 'mean')]
    std_sales = game_releases_by_year_normal[('total_sales', 'std')]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))


    axs[0].plot(game_releases_by_year_normal.index, mean_sales, label='Mean total_sales', color='blue')
    axs[0].fill_between(game_releases_by_year_normal.index, mean_sales - std_sales, mean_sales + std_sales, alpha=0.3, color='blue')
    axs[0].set_title('Mean total sales of a game per Year with Std Dev')
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Sales (Millions)')
    axs[0].grid(True)
    axs[0].legend()


    # Right: Boxplot
    # Extract values from skewed summary DataFrame
    years = game_releases_by_year_skewed.index
    medians = game_releases_by_year_skewed[('total_sales', 'median')]
    q1 = game_releases_by_year_skewed[('total_sales', 'q1')]
    q3 = game_releases_by_year_skewed[('total_sales', 'q3')]
    iqr = q3 - q1

    # Construct the boxplot-compatible stats manually
    # Construct the boxplot-compatible stats manually
    box_data = []
    for median, q_1, q_3, iqr_val in zip(medians, q1, q3, iqr):
        # Compute "whiskers" (1.5 IQR rule)
        lower_whisker = max(q_1 - 1.5 * iqr_val, 0)
        upper_whisker = q_3 + 1.5 * iqr_val
        box_data.append({
            'med': median,
            'q1': q_1,
            'q3': q_3,
            'whislo': lower_whisker,
            'whishi': upper_whisker,
            'fliers': []
        })

    # Right: Custom boxplot from stats
    axs[1].bxp(box_data, positions=range(len(years)), showfliers=False, widths=0.6)
    axs[1].set_xticks(range(len(years)))
    axs[1].set_xticklabels([str(y) for y in years.year], rotation=45)
    axs[1].set_title('Boxplot of Total Sales per Year (Summary)')
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Sales (Millions)')
    axs[1].set_position([0.55, 0.1, 0.4, 0.8])  # [left, bottom, width, height]

    plt.tight_layout()
    plt.show()

def sales_by_platform_and_year(df):
    agg_dict = {
    'name': 'count',
    'platform': 'count',
    'genre': 'count',
    'na_sales': 'sum',
    'eu_sales': 'sum',
    'jp_sales': 'sum',
    'other_sales': 'sum',
    'critic_score': 'mean',
    'user_score': 'mean',
    'rating': 'count',
    'total_sales': 'sum'
}
    game_releases_by_platform_year = df.groupby(['platform','year_of_release']).agg(agg_dict)
    game_releases_by_platform_year.head()

    # Heatmap of platform sales over time
    # new df for heatmap
    heatmap_data = df.groupby(['platform', 'year_of_release'])['total_sales'].sum().reset_index()

    # Pivot for the heatmap
    heatmap_data = heatmap_data.pivot(
        index='platform',
        columns='year_of_release',
        values='total_sales'
    )

    # the heatmap
    plt.figure(figsize=(15, 10))
    plt.imshow(heatmap_data, cmap='YlGnBu', interpolation='nearest')

    # color bar
    plt.colorbar()

    # Set labels and title
    plt.xlabel('Year')
    plt.ylabel(heatmap_data.index.name)
    plt.title('Heatmap of Total Sales by Platform and Year')

    # Set tick labels on the axes
    plt.xticks(
        ticks=np.arange(len(heatmap_data.columns)),
        labels=heatmap_data.columns.year,
        rotation=45,
        ha='right'
    )
    plt.yticks(
        ticks=np.arange(len(heatmap_data.index)),
        labels=heatmap_data.index
    )

    # Show the plot
    plt.show()

def relevant_period(df, start_year=2013, end_year=2016):
    relevant_years = list(range(start_year, end_year)) 
    df_relevant = df[df['year_of_release'].dt.year.isin(relevant_years)]
    print(df_relevant.head())

    # platform sales trends
    sales_last_3_years = df_relevant.groupby(['platform', 'year_of_release'])['total_sales'].sum() #series
    print(df_relevant)

    # Sort platforms by total sales
    sales_last_3_years = sales_last_3_years.reset_index()
    platform_sales_last_3_years = sales_last_3_years.groupby('platform')['total_sales'].sum()
    platform_sales_last_3_years.sort_values( ascending = False)

    # Visualize top platforms
    platform_sales_last_3_years = platform_sales_last_3_years.reset_index()

    # Preparing heatmap data by setting 'platform' as the index
    heatmap_data = platform_sales_last_3_years.set_index('platform')[['total_sales']].sort_values(by='total_sales', ascending = False)

    # heatmap
    plt.figure(figsize=(15, 3))
    plt.imshow(heatmap_data, cmap='YlGnBu', interpolation='nearest')

    # color bar
    plt.colorbar()

    # Set labels and title
    plt.xlabel('Metric')
    plt.ylabel('Platform')
    plt.title('Heatmap of Total Sales by Platform')


    plt.xticks(
        ticks=np.arange(heatmap_data.shape[1]),
        labels=heatmap_data.columns,
        rotation=0,
        ha='right'
    )

    plt.yticks(
        ticks=np.arange(heatmap_data.shape[0]),
        labels=heatmap_data.index
    )

    # Show the plot
    plt.show()

    # year-over-year growth for each platform
    sales_last_3_years['sales_growth'] = (sales_last_3_years['total_sales'] / sales_last_3_years.groupby('platform')['total_sales'].shift(1)) - 1
    sales_last_3_years['sales_growth'] = sales_last_3_years['sales_growth'].replace([np.inf, -np.inf], np.nan)


    # year-over-year growth
    heatmap_data = sales_last_3_years.reset_index()
    heatmap_data = heatmap_data.pivot(
        index='platform',
        columns='year_of_release',
        values='sales_growth'
    )

    # heatmap
    plt.figure(figsize=(15, 10))
    plt.imshow(heatmap_data, cmap='YlGnBu', interpolation='nearest')

    # color bar
    plt.colorbar()

    # Set labels and title
    plt.xlabel('Year')
    plt.ylabel(heatmap_data.index.name)
    plt.title('Heatmap of Sales Growth by Platform and Year')

    plt.xticks(
        ticks=np.arange(len(heatmap_data.columns)),
        labels=heatmap_data.columns.year,
        rotation=45,
        ha='right'
    )
    plt.yticks(
        ticks=np.arange(len(heatmap_data.index)),
        labels=heatmap_data.index
    )

    # Show the plot
    plt.show()

    # statistics for each platform
    print(sales_last_3_years.describe())
    print(sales_last_3_years.groupby('platform')['total_sales'].median())

    # scatter plots for both critic and user scores
    scatterplot_data = df_relevant[['critic_score','user_score']].dropna()
    plt.scatter(scatterplot_data['critic_score'], scatterplot_data['user_score'], alpha=0.5)
    plt.xlabel(df['critic_score'].name) 
    plt.ylabel(df['user_score'].name)    
    plt.title('Scatter Plot of Critic Score vs. User Score')
    plt.show()

    return df_relevant

def relationship(cols):
    scores_df = cols.dropna()

    rho, p_value = st.spearmanr(scores_df.iloc[:,0], scores_df.iloc[:,1])
    print(f'correlation: {rho}\np-value: {p_value}')

def last_3_years_df(df):
    return df[df['year_of_release'].dt.year >= (df['year_of_release'].dt.year.max() - 9)]

def n_unique_platforms(df):
     return df.groupby(['name'])['platform'].nunique() 

def name_platform_total_sales(df):
    return df.groupby(['name', 'platform'])['total_sales'].sum().reset_index()

def compare_name_total_sales_x_platform(df):
    compare_name_total_sales_x_platform = (
        df.pivot(
            index = 'name',
            columns = 'platform',
            values = 'total_sales'
        )
    )
    compare_name_total_sales_x_platform = compare_name_total_sales_x_platform.fillna(value = 0)
    print(compare_name_total_sales_x_platform.head())
    return compare_name_total_sales_x_platform

def genre_performance(df):
    genre =  df.groupby('genre')['total_sales'].sum().sort_values(ascending = False)


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
    return genre

def market_share(df_relevant,genre):
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
    year_genre['average_genre_sales_year'] = year_genre.mean(axis=1)
    print("Calculated average_genre_sales_year")

    #total sales of each genre across the years
    # subtract the average column we just added to avoid double counting if summing across axis 1, 
    # but here we are adding a ROW 'total_sales' which is the sum of columns.
    # We should exclude the 'average_genre_sales_year' column from the sum if we want just the year sums.
    year_genre.loc['total_sales'] = year_genre.drop('average_genre_sales_year', axis=1).sum()
    print("Calculated total_sales row")

    #average sales
    year_genre.loc['average_sales'] = year_genre.drop('total_sales', axis=0).mean(axis=0)
    print("Calculated average_sales row")

    return year_genre

def performance(platform, region, df):
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

    performance = df.groupby(['platform'])[region].agg(agg_dict[region])
    print(f'{platform}:',performance.loc[platform])


def comparative_platform_analysis(df_relevant):
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
    print(platform_sales)

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

    return platform_sales

def genre_performance_by_region(df_relevant):
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

    genre_performance_region = df_relevant.groupby(['genre'])[['na_sales','eu_sales','jp_sales','other_sales']].agg(agg_dict)
    print(genre_performance_region)

    # heatmap
    plt.figure(figsize=(15, 10))
    plt.imshow(genre_performance_region, cmap='YlGnBu', interpolation='nearest')

    #  color bar
    plt.colorbar()

    # Set labels and title
    plt.xlabel('Platform')
    plt.ylabel(genre_performance_region.index.name)
    plt.title('Heatmap of Platform Sales by Region')

    plt.xticks(
        ticks=np.arange(len(genre_performance_region.columns)),
        labels=genre_performance_region.columns,
        rotation=45,
        ha='right'
    )
    plt.yticks(
        ticks=np.arange(len(genre_performance_region.index)),
        labels=genre_performance_region.index
    )

    # Show plot
    plt.show()

def esrb_impact(df_relevant):
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
    return rating_performance
