'''
objective: which of the plans brings in more revenue in order to adjust the advertising budget?
'''

# Loading all the libraries
import pandas as pd		
import numpy as np	
from math import factorial	
import matplotlib.pyplot as plt
from scipy import stats as st

# Load the data files into different DataFrames
calls_df = pd.read_csv("/datasets/megaline_calls.csv")
internet_df = pd.read_csv("/datasets/megaline_internet.csv")
messages_df = pd.read_csv("/datasets/megaline_messages.csv")
plans_df = pd.read_csv("/datasets/megaline_plans.csv")
users_df = pd.read_csv("/datasets/megaline_users.csv")

# save as dictionary
dfs ={
    'calls_df':calls_df,

'internet_df':internet_df,

'messages_df' :messages_df ,

'plans_df' :plans_df ,

'users_df': users_df
}

# EDA
# Print the general/summary information about the plans' DataFrame
# this is the start of a function for my toolbox
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

#view_raw_df(dfs)
[view_raw_df(df) for df_name, df in dfs.items()]

print(plans_df.info())

# Print a sample of data for plans
print(plans_df.sample(2, random_state = 1))

# fix data
# based on EDA
calls_df['id']= calls_df['id'].astype(str)
calls_df['user_id']= calls_df['user_id'].astype(str)
internet_df['id']= internet_df['id'].astype(str)
internet_df['user_id']= internet_df['user_id'].astype(str)
messages_df['id']= messages_df['id'].astype(str)
messages_df['user_id']= messages_df['user_id'].astype(str)
users_df['user_id']= users_df['user_id'].astype(str)

#convert datetime var to datetime object per df
print(calls_df['call_date'][0])
calls_df['call_date'] = pd.to_datetime(calls_df['call_date'], format = '%Y-%m-%d')
print((calls_df['call_date'].dtype))

print(internet_df['session_date'][0])
internet_df['session_date'] = pd.to_datetime(internet_df['session_date'], format = '%Y-%m-%d')
print(internet_df['session_date'].dtype)

print(messages_df['message_date'][0])
messages_df['message_date'] = pd.to_datetime(messages_df['message_date'], format = '%Y-%m-%d')
print(messages_df['message_date'].dtype)

print(users_df['reg_date'][0])
users_df['reg_date'] = pd.to_datetime(users_df['reg_date'],format = '%Y-%m-%d')
print(users_df['reg_date'].dtype)

print(users_df['churn_date'][281])
users_df['churn_date'] = pd.to_datetime(users_df['churn_date'],format = '%Y-%m-%d')
print(users_df['churn_date'].dtype)

#convert to year-month format 
calls_df['call_date'] = calls_df['call_date'].dt.to_period('M')
internet_df['session_date'] = internet_df['session_date'].dt.to_period('M')
messages_df['message_date'] = messages_df['message_date'].dt.to_period('M')
users_df['reg_date'] = users_df['reg_date'].dt.to_period('M')
users_df['churn_date'] = users_df['churn_date'].dt.to_period('M')

print(calls_df['call_date'].dtype)
print(internet_df['session_date'].dtype)
print(messages_df['message_date'].dtype)
print(users_df['reg_date'].dtype)
print(users_df['churn_date'].dtype)

#fillna
print(users_df['churn_date'].isnull().sum())
users_df['churn_date'].fillna(pd.NaT,inplace=True)
print(users_df['churn_date'].isnull().sum())
view_raw_df(users_df)

#It turns out it's already marked as NaT. I need a date to calculate revenue, so going to use max date of their activity
calls_date_max = calls_df.groupby('user_id')['call_date'].max().rename("call_date") #series
internet_date_max = internet_df.groupby('user_id')['session_date'].max().rename("session_date") #series
messages_date_max = messages_df.groupby('user_id')['message_date'].max().rename("message_date") #series

print(type(calls_date_max.index[0])) #str
print(type(internet_date_max.index[0])) #str
print(type(messages_date_max.index[0])) #str

calls_date_max = calls_date_max.sort_index()
internet_date_max = internet_date_max.sort_index()
messages_date_max = messages_date_max.sort_index()

print(calls_date_max.index) #df looks good
print(internet_date_max.index) #df looks good
print(messages_date_max.index) #df looks good

# they don't have a common index, so need to create an index
activity_dates = pd.concat([calls_date_max, internet_date_max, messages_date_max], axis=1, join = 'outer') #df ✅
activity_dates = activity_dates.dropna(how='any') #required so that max doesn't return NaN
user_max_activity = activity_dates.max(axis = 1) #series
print(activity_dates.head())
print(users_max_activity.head())

# fix users_df
users_df['user_id']= users_df['user_id'].astype(str)
print(users_df['churn_date'].isnull().sum())
users_df['churn_date'].fillna(users_df['user_id'].map(user_max_activity),inplace = True) #fillna with max date via user_id
print(users_df['churn_date'].isnull().sum())
users_df['churn_date'].fillna(user_max_activity.max(),inplace = True) #fillna with max over all date
print(users_df['churn_date'].isnull().sum())

#remove any duplicates
print(calls_df.duplicated().sum())
print(internet_df.duplicated().sum())
print(messages_df.duplicated().sum())
print(plans_df.duplicated().sum())
print(users_df.duplicated().sum())

#none found

# enrich data
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

#do this for all datetime cols
append_datetime_features(calls_df,['call_date'])
append_datetime_features(internet_df,['session_date'])
append_datetime_features(messages_df,['message_date'])
append_datetime_features(users_df,['reg_date'])
append_datetime_features(users_df,['churn_date'])

#calls_df	call_date ✅
#internet_df	session_date ✅
#messages_df	message_date ✅
#users_df	reg_date ✅
#users_df	churn_date ✅

# Print the general/summary information about the users' DataFrame
view_raw_df(users_df)

# Print a sample of data for users
users_df.sample(5, random_state = 1)

# Print the general/summary information about the calls' DataFrame
view_raw_df(calls_df)

# Print a sample of data for calls
calls_df.sample(5, random_state = 1)

# Print the general/summary information about the messages' DataFrame
view_raw_df(messages_df)

# Print a sample of data for messages
messages_df.sample(5, random_state = 1)

# Print the general/summary information about the internet DataFrame
view_raw_df(internet_df)

# Print a sample of data for the internet traffic
internet_df.sample(5, random_state = 1)

# Print out the plan conditions and make sure they are clear for you
print(plans_df)

#There are two plans, ultimate costs more than surf, but comes with more of everything and at a cheaper unit price.
# revenue/user = calls + messages + internet usage

# Calculate the number of calls made by each user per month. Save the result.
agg_dict = {'id':'count',
            'call_date':'max',
            'duration':'sum',
            '_call_date_year':'count',
           }
user_calls_by_month = calls_df.groupby(['user_id','_call_date_month']).agg(agg_dict)

#view_raw_df(user_calls_by_month) reformat
print(user_calls_by_month.head(1))
user_calls_by_month.reset_index(inplace = True)
print(user_calls_by_month.head(1))
user_calls_by_month.rename(columns = {"_call_date_month": "month"}, inplace = True)
print(user_calls_by_month.head(1))

# Calculate the amount of minutes spent by each user per month. Save the result.
print(user_calls_by_month['duration'].sample(10,random_state=1))
print(user_calls_by_month.head())

# Calculate the number of messages sent by each user per month. Save the result.
agg_dict = {'id':'count',
            'message_date':'max',
            '_message_date_year':'count',
           }
user_messages_by_month = messages_df.groupby(['user_id','_message_date_month']).agg(agg_dict)
user_messages_by_month.reset_index(inplace = True)
user_messages_by_month.rename(columns = {"_message_date_month":"month"}, inplace = True)
print(user_messages_by_month['id'].sample(10,random_state=1))
print(user_messages_by_month.head())

# Calculate the volume of internet traffic used by each user per month. Save the result.
agg_dict = {'id':'count',
            'session_date':'max',
            'mb_used': 'sum',
            '_session_date_year':'count',
           }
user_traffic_by_month = internet_df.groupby(['user_id','_session_date_month']).agg(agg_dict)
print(user_traffic_by_month['id'].sample(10,random_state=1))
print(user_traffic_by_month['mb_used'].sample(10,random_state=1))
user_traffic_by_month.reset_index(inplace = True)
user_traffic_by_month.rename(columns = {'_session_date_month':'month'},inplace = True)
print(user_traffic_by_month)

# Merge the data for calls, minutes, messages, internet based on user_id and month
user_calls_messages_traffic_by_month = (
    user_calls_by_month
    .merge(user_messages_by_month,
           left_on=['user_id','month'],
           right_on=['user_id','month'],
           how = 'left',
           suffixes = ("","_messages")
    )
    .merge(user_traffic_by_month,
           left_on=['user_id','month'],
           right_on=['user_id','month'],
           how = 'left',
           suffixes = ("",'_traffic')
           )
)
view_raw_df(user_calls_messages_traffic_by_month)

# Add the plan information
user_by_plan = (
    users_df
    .merge(
        plans_df,
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

# compare plans
#group by plan
# sum
plan_revenue = user_calls_messages_traffic_plan_by_month[['plan', 'call_count', 'duration', 'message_count', 'session_count', 'mb_used','revenue']]
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

# mean
agg_dict={ 
    'call_count': 'mean', 
    'duration': 'mean', 
    'message_count': 'mean',
    'session_count': 'mean', 
    'mb_used': 'mean', 
    'revenue': 'mean'
}
mean_plan_revenue = plan_revenue.groupby('plan').agg(agg_dict)
print(mean_plan_revenue.head())

#plot
for col in sum_plan_revenue.columns[1:]:
    sum_plan_revenue[col].plot(
        kind='bar',
        title=f'{col} by Plan (sum)',
        figsize=(12, 6),
        ylabel=col
    )
    plt.xticks(rotation=45, ha='right') #tilts text 45, horizontal alignment set to right
    plt.show()

for col in mean_plan_revenue.columns[1:]:
    mean_plan_revenue[col].plot(
        kind='bar',
        title=f'{col} by Plan (mean)',
        figsize=(12, 6),
        ylabel=col
    )
    plt.xticks(rotation=45, ha='right') #tilts text 45, horizontal alignment set to right
    plt.show()

# Compare the number of minutes users of each plan require each month. 
print(user_calls_messages_traffic_plan_by_month.columns)
#metrics = ['plan', 'id_calls', 'duration', 'id_messages', 'id_traffic', 'mb_used', 'revenue']
metrics = ['duration']

# Get unique months
unique_months = user_calls_messages_traffic_plan_by_month['month'].unique()

# histograms for each month
for month in unique_months:
    month_data = user_calls_messages_traffic_plan_by_month[user_calls_messages_traffic_plan_by_month['month'] == month]
    
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
for each in user_calls_messages_traffic_plan_by_month['plan'].unique():
    print(f' {each} mean: ',user_calls_messages_traffic_plan_by_month[user_calls_messages_traffic_plan_by_month['plan'] == each]['duration'].mean())
    print(f' {each} variance: ',user_calls_messages_traffic_plan_by_month[user_calls_messages_traffic_plan_by_month['plan'] == each]['duration'].var())

# boxplot to visualize the distribution of the monthly call duration
user_calls_messages_traffic_plan_by_month.boxplot(column='duration', by='plan_name', grid=False)

# number of messages users of each plan tend to send each month
user_calls_messages_traffic_plan_by_month.boxplot(column='message_count', by='plan_name', grid=False)

# amount of internet traffic consumed by users per plan
user_calls_messages_traffic_plan_by_month.boxplot(column='session_count', by='plan_name', grid=False)

# revenue per plan
user_calls_messages_traffic_plan_by_month.boxplot(column='revenue', by='plan_name', grid=False)

# statistical difference
results = st.ttest_ind(user_calls_messages_traffic_plan_by_month[user_calls_messages_traffic_plan_by_month['plan_name'] == 'surf']['revenue'].dropna(), user_calls_messages_traffic_plan_by_month[user_calls_messages_traffic_plan_by_month['plan_name'] == 'ultimate']['revenue'].dropna(), equal_var = False)
print(results)

#user_calls_messages_traffic_plan_by_month.columns
user_calls_messages_traffic_plan_by_month['city'].unique()

# Test specific hypotheses by client request
#H0: the means are the same between the NY-NJ area and that of the users from the other regions.
#H1: the means are not the same. 
#alpha = 0.05

results_2 = (
    st.ttest_ind(
        user_calls_messages_traffic_plan_by_month[user_calls_messages_traffic_plan_by_month['city'] == 'New York-Newark-Jersey City, NY-NJ-PA MSA']['revenue'].dropna(),
        user_calls_messages_traffic_plan_by_month[user_calls_messages_traffic_plan_by_month['city'] != 'New York-Newark-Jersey City, NY-NJ-PA MSA']['revenue'].dropna(),
        equal_var = False))
print(results_2)