'''
objective: which of the plans brings in more revenue in order to adjust the advertising budget?
'''

# Loading all the libraries	
import matplotlib.pyplot as plt
from scipy import stats as st

from src.data_preprocessing import load_data, view_raw_df, append_datetime_features
from src.data_preprocessing import set_datatype, fill_missing, deduplicate
from src.data_preprocessing import n_monthly_calls, n_monthly_messages, n_monthly_internet, monthly_revenue, sum_plan_revenue, mean_plan_revenue, plot_revenue, monthly_minutes_by_plan

# python.exe -m pip install --upgrade pip

#paths
calls = 'data/megaline_calls.csv'
internet = 'data/megaline_internet.csv'
messages = 'data/megaline_messages.csv'
plans = 'data/megaline_plans.csv'
users = 'data/megaline_users.csv'

# Load the data files into different DataFrames
calls_df = load_data(calls)
internet_df = load_data(internet)
messages_df = load_data(messages)
plans_df = load_data(plans)
users_df = load_data(users)

# save as dictionary
dfs ={
    'calls_df':calls_df,
'internet_df':internet_df,
'messages_df' :messages_df ,
'plans_df' :plans_df ,
'users_df': users_df
}

#view_raw_df(dfs)
[view_raw_df(df) for df in dfs.values()]

# plans_df
print(plans_df.info())

# Print a sample of data for plans
print(plans_df.sample(2, random_state = 1))

# fix data
# based on EDA
calls_df['id'] = set_datatype(calls_df['id'], 'str')
calls_df['user_id'] = set_datatype(calls_df['user_id'], 'str')
internet_df['id'] = set_datatype(internet_df['id'], 'str')
internet_df['user_id'] = set_datatype(internet_df['user_id'], 'str')
messages_df['id'] = set_datatype(messages_df['id'], 'str')
messages_df['user_id'] = set_datatype(messages_df['user_id'], 'str')
users_df['user_id'] = set_datatype(users_df['user_id'], 'str')


#convert datetime var to datetime object year-month format
calls_df['call_date'] = set_datatype(calls_df['call_date'])
internet_df['session_date'] = set_datatype(internet_df['session_date'])
messages_df['message_date'] = set_datatype(messages_df['message_date'])
users_df['reg_date'] = set_datatype(users_df['reg_date'])
users_df['churn_date'] = set_datatype(users_df['churn_date'])
users_df['user_id'] = set_datatype(users_df['user_id'], 'str')

#fillna
fill_missing(users_df,['churn_date'])
fill_missing(calls_df,['call_date'])
fill_missing(internet_df,['session_date'])
fill_missing(messages_df,['message_date'])

# deduplicate
calls_df = deduplicate(calls_df)
internet_df = deduplicate(internet_df)
messages_df = deduplicate(messages_df)
users_df = deduplicate(users_df)
plans_df = deduplicate(plans_df)

#do this for all datetime cols
print(f'dtype: {calls_df['call_date'].dtype}') #object
append_datetime_features(calls_df,['call_date'])
append_datetime_features(internet_df,['session_date'])
append_datetime_features(messages_df,['message_date'])
append_datetime_features(users_df,['reg_date'])
append_datetime_features(users_df,['churn_date'])

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
user_calls_by_month = n_monthly_calls(calls_df)

# Calculate the number of messages sent by each user per month. Save the result.
user_messages_by_month = n_monthly_messages(messages_df)

# Calculate the volume of internet traffic used by each user per month. Save the result.
user_traffic_by_month = n_monthly_internet(internet_df)

# calculate revenue per user per month
user_calls_messages_traffic_plan_by_month = monthly_revenue(
    user_calls_by_month,
    user_messages_by_month,
    user_traffic_by_month,
    users_df,
    plans_df
)

# compare plans
#group by plan
# sum
sum_plan_revenue_results, plan_revenue = sum_plan_revenue(user_calls_messages_traffic_plan_by_month)

# mean
mean_plan_revenue_results = mean_plan_revenue(plan_revenue)

#plot
plot_revenue(sum_plan_revenue_results, mean_plan_revenue_results)

# Compare the number of minutes users of each plan require each month. 
# Test specific hypotheses by client request
    #H0: the means are the same between the NY-NJ area and that of the users from the other regions.
    #H1: the means are not the same. 
    #alpha = 0.05
results, nj_results = monthly_minutes_by_plan(user_calls_messages_traffic_plan_by_month)