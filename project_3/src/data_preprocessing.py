import pandas as pd		
import numpy as np    
import matplotlib.pyplot as plt
from scipy import stats as st


def load_data(path):
    return pd.read_csv(path)

def view_raw_df(dfs):
    if isinstance(dfs, pd.DataFrame):
        dfs = {'df': dfs}  # Wrap single DataFrame in a dict with a default name

    summaries = []
    for df_name, df in dfs.items():
        for col in df.columns:
            common_unique_values = df[col].value_counts().head(30).index.tolist()
            rare_unique_values = df[col].value_counts(sort=False).head(30).index.tolist()
            data_type = type(df[col].iloc[0])
            series_count = df[col].count()
            missing_values = len(df) - series_count

            summaries.append({
                'DataFrame': f'{df_name}',
                'Column': col,
                'Common Unique Values': common_unique_values,
                'Rare Unique Values': rare_unique_values,
                'Data Type': data_type,
                'Series Count': series_count,
                'Missing Values': missing_values
            })

    summary_df = pd.DataFrame(summaries) # Python converts keys as col names, value as row values
  
# missing data may affect data type, so make sure all the values are labelled and logical
# var with fake numbers should be string: zip codes, ID numbers, Usernames, Addresses, phone numbers   
    
    return summary_df

def append_datetime_features(df, col_name):
    col = df[str(col_name[0])]
    # Choose which features you want to include
    choose_your_features = [
        'year',
        'month',
#         'day',
#         'hour',
#         'minute',
#         'second',
#         'dayofweek',
#         'dayofyear',
#         'weekofyear',
#         'quarter',
#         'is_month_start',
#         'is_month_end',
#         'is_quarter_start',
#         'is_quarter_end',
#         'is_year_start',
#         'is_year_end',
#         'date',
#         'time'
    ] #18 originally
    
    # Validate that input is a datetime series
    if not pd.api.types.is_datetime64_any_dtype(col):
        raise ValueError("Input column must be pd.Series datetime dtype.")
    
    for feature in choose_your_features:
        try:
            df[f"_{col_name[0]}_{feature}"] = getattr(col.dt, feature)
            
        except AttributeError:
            raise ValueError(f"'{feature}' is not a valid datetime accessor property.")

    return df.info()

def set_datatype(column, dtype=None):
    try:
        if dtype is None:
            # Default behavior: try converting to datetime
            return pd.to_datetime(column, errors="coerce")
        elif isinstance(dtype, str) and "datetime" in dtype.lower():
            return pd.to_datetime(column, errors="coerce")
        else:
            return column.astype(dtype)
    except Exception as e:
        print(f"Error converting column to {dtype}: {e}")
        return column
    
def fill_missing(df = None, column = None):
    if column == 'churn_date':
        df[column].fillna(pd.NaT,inplace=True)
        # get max date for each user in each activity df
        activity_dates = pd.concat([calls_date_max, internet_date_max, messages_date_max], axis=1, join = 'outer') #df ✅
        activity_dates = activity_dates.dropna(how='any') 
        user_max_activity = activity_dates.max(axis = 1)
        df[column].fillna(df['user_id'].map(user_max_activity),inplace = True) #fillna with max date via user_id
        df[column].fillna(user_max_activity.max(),inplace = True) #fillna with max over all date
    
    elif column == 'call_date':
        calls_date_max = df.groupby('user_id')['call_date'].max().rename("call_date").sort_index() #series
    
    elif column == 'session_date':
        internet_date_max = df.groupby('user_id')['session_date'].max().rename("session_date").sort_index() #series
    
    elif column == 'message_date':
        messages_date_max = df.groupby('user_id')['message_date'].max().rename("message_date").sort_index() #series

def deduplicate(df):
    return df.drop_duplicates()

def n_monthly_calls(df):
    # Calculate the number of calls made by each user per month. Save the result.
    agg_dict = {'id':'count',
                'call_date':'max',
                'duration':'sum',
                '_call_date_year':'count',
            }
    user_calls_by_month = df.groupby(['user_id','_call_date_month']).agg(agg_dict)

    #view_raw_df(user_calls_by_month) reformat
    user_calls_by_month.reset_index(inplace = True)
    user_calls_by_month.rename(columns = {"_call_date_month": "month"}, inplace = True)

    # Calculate the amount of minutes spent by each user per month. Save the result.
    print(user_calls_by_month['duration'].sample(10,random_state=1))
    print(user_calls_by_month.head())
    return user_calls_by_month

def n_monthly_messages(df):
    agg_dict = {'id':'count',
            'message_date':'max',
            '_message_date_year':'count',
           }
    user_messages_by_month = df.groupby(['user_id','_message_date_month']).agg(agg_dict)
    user_messages_by_month.reset_index(inplace = True)
    user_messages_by_month.rename(columns = {"_message_date_month":"month"}, inplace = True)
    print(user_messages_by_month['id'].sample(10,random_state=1))
    print(user_messages_by_month.head())
    return user_messages_by_month

def n_monthly_internet(df):
    agg_dict = {'id':'count',
            'session_date':'max',
            'mb_used': 'sum',
            '_session_date_year':'count',
           }
    user_traffic_by_month = df.groupby(['user_id','_session_date_month']).agg(agg_dict)
    print(user_traffic_by_month['id'].sample(10,random_state=1))
    print(user_traffic_by_month['mb_used'].sample(10,random_state=1))
    user_traffic_by_month.reset_index(inplace = True)
    user_traffic_by_month.rename(columns = {'_session_date_month':'month'},inplace = True)
    print(user_traffic_by_month)
    return user_traffic_by_month

def monthly_revenue(calls, messages, internet, users, plans):
    # Merge the data for calls, minutes, messages, internet based on user_id and month
    user_calls_messages_traffic_by_month = (
        calls
        .merge(messages,
            left_on=['user_id','month'],
            right_on=['user_id','month'],
            how = 'left',
            suffixes = ("","_messages")
        )
        .merge(internet,
            left_on=['user_id','month'],
            right_on=['user_id','month'],
            how = 'left',
            suffixes = ("",'_traffic')
            )
    )
    view_raw_df(user_calls_messages_traffic_by_month)

    # Add the plan information
    user_by_plan = (
        users
        .merge(
            plans,
            left_on = 'plan',
            right_on = 'plan_name',
            how = 'left',
            suffixes = ('','_plans')
        )
    )
    user_calls_messages_traffic_plan_by_month = (
        user_calls_messages_traffic_by_month
        .merge(
            user_by_plan,
            left_on = ['user_id'],
            right_on = ['user_id'],
            how = 'left',
            suffixes = ('','_plan')
        )
    )
    view_raw_df(user_calls_messages_traffic_plan_by_month)
    print(user_calls_messages_traffic_plan_by_month.head(5))

    #calculate revenue per user per month
    user_calls_messages_traffic_plan_by_month['revenue'] = (
        user_calls_messages_traffic_plan_by_month['usd_monthly_pay'] + 
        user_calls_messages_traffic_plan_by_month['usd_per_gb'] * ((user_calls_messages_traffic_plan_by_month['mb_used'] - user_calls_messages_traffic_plan_by_month['messages_included']) / 1000).clip(lower=0) + 
        user_calls_messages_traffic_plan_by_month['usd_per_message'] * (user_calls_messages_traffic_plan_by_month['id_messages'] - user_calls_messages_traffic_plan_by_month['messages_included']).clip(lower=0) + 
        user_calls_messages_traffic_plan_by_month['usd_per_minute'] * (user_calls_messages_traffic_plan_by_month['duration'] - user_calls_messages_traffic_plan_by_month['minutes_included']).clip(lower=0)
    )

    #rename the id variables
    user_calls_messages_traffic_plan_by_month.rename(columns = {'id':'call_count','id_messages':'message_count','id_traffic':'session_count'}, inplace = True)
    print(user_calls_messages_traffic_plan_by_month.head())

    return user_calls_messages_traffic_plan_by_month

def sum_plan_revenue(df):
    plan_revenue = df[['plan', 'call_count', 'duration', 'message_count', 'session_count', 'mb_used','revenue']]
    agg_dict={ 
        'call_count': 'sum', 
        'duration': 'sum', 
        'message_count': 'sum',
        'session_count': 'sum', 
        'mb_used': 'sum', 
        'revenue': 'sum'
    }
    sum_plan_revenue = plan_revenue.groupby('plan').agg(agg_dict)
    print(sum_plan_revenue.head())
    return sum_plan_revenue, plan_revenue

def mean_plan_revenue(df):
    agg_dict={ 
    'call_count': 'mean', 
    'duration': 'mean', 
    'message_count': 'mean',
    'session_count': 'mean', 
    'mb_used': 'mean', 
    'revenue': 'mean'
    }
    mean_plan_revenue = df.groupby('plan').agg(agg_dict)
    print(mean_plan_revenue.head())
    return mean_plan_revenue

def plot_revenue(sum, mean):
    for col in sum.columns[1:]:
        sum[col].plot(
            kind='bar',
            title=f'{col} by Plan (sum)',
            figsize=(12, 6),
            ylabel=col
        )
        plt.xticks(rotation=45, ha='right') #tilts text 45, horizontal alignment set to right
        plt.show()

    for col in mean.columns[1:]:
        mean[col].plot(
            kind='bar',
            title=f'{col} by Plan (mean)',
            figsize=(12, 6),
            ylabel=col
        )
        plt.xticks(rotation=45, ha='right') #tilts text 45, horizontal alignment set to right
        plt.show()

def monthly_minutes_by_plan(df):
    print(df.columns)
    #metrics = ['plan', 'id_calls', 'duration', 'id_messages', 'id_traffic', 'mb_used', 'revenue']
    metrics = ['duration']

    # Get unique months
    unique_months = df['month'].unique()

    # histograms for each month
    for month in unique_months:
        month_data = df[df['month'] == month]
        
        for metric in metrics:
            plt.figure(figsize=(8, 5))
            month_data[metric].hist(bins=30, alpha=0.7)
            plt.title(f'{metric} distribution in Month {month}')
            plt.xlabel(metric)
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # mean and the variance of the monthly call duration
    for each in df['plan'].unique():
        print(f' {each} mean: ',df[df['plan'] == each]['duration'].mean())
        print(f' {each} variance: ',df[df['plan'] == each]['duration'].var())

    # boxplot to visualize the distribution of the monthly call duration
    df.boxplot(column='duration', by='plan_name', grid=False)

    # number of messages users of each plan tend to send each month
    df.boxplot(column='message_count', by='plan_name', grid=False)

    # amount of internet traffic consumed by users per plan
    df.boxplot(column='session_count', by='plan_name', grid=False)

    # revenue per plan
    df.boxplot(column='revenue', by='plan_name', grid=False)

    # statistical difference
    results = st.ttest_ind(df[df['plan_name'] == 'surf']['revenue'].dropna(), df[df['plan_name'] == 'ultimate']['revenue'].dropna(), equal_var = False)
    print(f'surf vs ultimate pvalue: {results.pvalue}')

    #user_calls_messages_traffic_plan_by_month.columns
    df['city'].unique()

    # Test specific hypotheses by client request
    #H0: the means are the same between the NY-NJ area and that of the users from the other regions.
    #H1: the means are not the same. 
    #alpha = 0.05

    nj_results = (
        st.ttest_ind(
            df[df['city'] == 'New York-Newark-Jersey City, NY-NJ-PA MSA']['revenue'].dropna(),
            df[df['city'] != 'New York-Newark-Jersey City, NY-NJ-PA MSA']['revenue'].dropna(),
            equal_var = False))
    print(f'New York-Newark-Jersey City, NY-NJ-PA MSA vs other regions pvalue: {nj_results.pvalue}')
    return results, nj_results
