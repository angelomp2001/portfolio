# Loading all the libraries	
import matplotlib.pyplot as plt
from scipy import stats as st

from src.data_preprocessing import load_all_data, inspect_initial_data, view_raw_df, append_datetime_features
from src.data_preprocessing import set_datatype, fill_missing, deduplicate
from src.data_preprocessing import prepare_data_types, handle_missing_values, remove_duplicates, enrich_data
from src.data_preprocessing import n_monthly_calls, n_monthly_messages, n_monthly_internet, monthly_revenue, sum_plan_revenue, mean_plan_revenue, plot_revenue, monthly_minutes_by_plan

# Load the data files into different DataFrames
dfs = load_all_data()
inspect_initial_data(dfs)

# Process data (Fix types, fill missing, deduplicate, enrich)
dfs = prepare_data_types(dfs)
dfs = handle_missing_values(dfs)
dfs = remove_duplicates(dfs)
dfs = enrich_data(dfs)

# Unpack DataFrames for further processing
calls_df = dfs['calls_df']
internet_df = dfs['internet_df']
messages_df = dfs['messages_df']
plans_df = dfs['plans_df']
users_df = dfs['users_df']

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

# Final Conclusions based on statistical tests
print("\n--- Final Conclusion ---")
print("Objective: which of the plans brings in more revenue in order to adjust the advertising budget?")
print(f"Surf vs Ultimate Revenue Difference p-value: {results.pvalue}")
if results.pvalue < 0.05:
    print("Result: Reject H0. There is a statistically significant difference in revenue between the plans.")
    # Assuming from EDA that Surf often has higher total revenue due to volume, but Ultimate has consistent higher per-user revenue.
    # The script output shows the means/totals to confirm which is 'more'.
else:
    print("Result: Fail to reject H0. No statistically significant difference found in revenue between plans.")

print(f"NY-NJ vs Other Regions Revenue Difference p-value: {nj_results.pvalue}")
if nj_results.pvalue < 0.05:
    print("Result: Reject H0. Geography (NY-NJ) significantly impacts revenue.")
else:
    print("Result: Fail to reject H0. Geography does not significantly impact revenue.")