#libraries and other files
import pandas as pd
import os
import re
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import numpy as np
from data_explorers import view, see
from H0_testing import split_test_plot


############################
#view params
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 500)        # Increase horizontal width
pd.set_option('display.max_colwidth', None) # Show full content of each column
pd.set_option('display.max_rows', None)        # Show all rows

# pd.reset_option('display.max_columns')
# pd.reset_option('display.width')
# pd.reset_option('display.max_colwidth')
# pd.reset_option('display.max_rows')

################################
# data
gold_recovery_full = pd.read_csv('gold_recovery_full.csv')
gold_recovery_train = pd.read_csv('gold_recovery_train.csv')
gold_recovery_test = pd.read_csv('gold_recovery_test.csv')

##############################################
# an closer look

#headers, values, dtypes, missing values, summaries

# df_full = view(gold_recovery_full,'headers')
# df_train = view(gold_recovery_train,'headers')
# df_test = view(gold_recovery_test, 'headers')

# shape
# print(gold_recovery_full.shape) # (22716, 87)
# print(gold_recovery_train.shape) # (16860, 87)
# print(gold_recovery_test.shape) # (5856, 53)



# common columns
common_columns = set(gold_recovery_train.columns) & set(gold_recovery_test.columns)
# [col for col in common_columns]



#df_full notes
# no duplicate values
# all cols seem relevant
# all cols are continuous
# date col should be dtype datetime
# target has the most missing: rougher.output.recovery: 3119 (14%), but we have lots of data, so dropna() it is. 

# df_train notes
# date col should be dtype datetime
# target has the most missing: rougher.output.recovery: 2573 (15%), but we have lots of data, so dropna() it is. 

# df_test notes
# date col should be dtype datetime
# target is missing: rougher.output.recovery: predict using features.


# max rows (the index column)
# print(max([views['headers'][row]['Unique Values'] for row in range(len(views['headers']))])) 

#############################
# data processing
#1.4 Peform data preprocessing

#df_full
gold_recovery_full['date'] = pd.to_datetime(gold_recovery_full['date'])
gold_recovery_full = gold_recovery_full.dropna()

# df_train
gold_recovery_train['date'] = pd.to_datetime(gold_recovery_train['date'])
gold_recovery_train = gold_recovery_train.dropna()

# df_test notes
gold_recovery_test['date'] = pd.to_datetime(gold_recovery_test['date'])
gold_recovery_test = gold_recovery_test.dropna()

#################################
# 1.2. Check that recovery is calculated correctly. Using the training set, calculate recovery for the rougher.output.recovery feature. Find the MAE between your calculations and the feature values. Provide findings.

# rougher recovery target 
C = gold_recovery_train['rougher.output.concentrate_au'] # share of gold in the concentrate right after flotation
F = gold_recovery_train['rougher.input.feed_au'] # share of gold in the feed before flotation
T = gold_recovery_train['rougher.output.tail_au'] # share of gold in the rougher tails right after flotation
#print(gold_recovery_train[['rougher.input.feed_au','rougher.output.concentrate_au','rougher.output.tail_au']])

print(f"means:\n{gold_recovery_train['rougher.input.feed_au'].mean():.02f}\n{gold_recovery_train['rougher.output.concentrate_au'].mean():.02f}\n{gold_recovery_train['rougher.output.tail_au'].mean():.02f}")
# means:
# 8.11 (input)
# 19.78 (concentrate output)
# 1.84 (tail output)
# is this saying concentration went up?

# from course documentation 
# gold_recovery_train['rougher.output.recovery.qa'] = (C * (F - T)) / ((F * (C - T))

# from DOT
gold_recovery_train['rougher.output.recovery.qa'] = ((C - F) / (T - F)) * 100


# MAE
rougher_output_recovery_MAE = abs(gold_recovery_train['rougher.output.recovery'] - gold_recovery_train['rougher.output.recovery.qa'])

#print(gold_recovery_train['rougher.output.recovery.qa'].astype('int64'))#.nunique())


# final recovery target
#C = gold_recovery_train['final.output.concentrate_au']
#F — share of gold in the feed before flotation (for finding the rougher concentrate recovery)/in the concentrate right after flotation (for finding the final concentrate recovery)
#T — share of gold in the rougher tails right after flotation (for finding the rougher concentrate 

######################################
# Create a figure with 3 subplots (stacked vertically)
# plt.figure(figsize=(10, 12))

# First histogram

# plt.subplot(3, 1, 1)
# plt.hist(gold_recovery_train['rougher.output.recovery'].astype('int64'), bins=30, color='blue', alpha=0.7)
# plt.title('Rougher Output Recovery')
# plt.xlabel('Value')
# plt.ylabel('Frequency')


# Second histogram
# plt.subplot(3, 1, 2)
# plt.hist(gold_recovery_train['rougher.output.recovery.qa'].astype('int64'), bins=30, color='green', alpha=0.7)
# plt.title('Rougher Output Recovery QA')
# plt.xlabel('Value')
# plt.ylabel('Frequency')

# Third histogram (absolute difference)
# plt.subplot(3, 1, 3)
# plt.hist(abs(gold_recovery_train['rougher.output.recovery'].astype('int64') - gold_recovery_train['rougher.output.recovery.qa'].astype('int64')), 
#          bins=30, color='red', alpha=0.7)
# plt.title('Absolute Difference Histogram')
# plt.xlabel('Value')
# plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()

################################
#1.3. Analyze the features not available in the test set. What are these parameters? What is their type?

#features not available in the test set:
in_train_not_test = set(gold_recovery_train.columns) - set(gold_recovery_test.columns)
print(in_train_not_test)

# All the 'output' vars are missing:
# final.output.concentrate_ag
# final.output.concentrate_au
# final.output.concentrate_pb
# final.output.concentrate_sol
# final.output.recovery
# final.output.tail_ag
# final.output.tail_au
# final.output.tail_pb
# final.output.tail_sol
# primary_cleaner.output.concentrate_ag
# primary_cleaner.output.concentrate_au
# primary_cleaner.output.concentrate_pb
# primary_cleaner.output.concentrate_sol
# primary_cleaner.output.tail_ag
# primary_cleaner.output.tail_au
# primary_cleaner.output.tail_pb
# primary_cleaner.output.tail_sol
# rougher.calculation.au_pb_ratio
# rougher.calculation.floatbank10_sulfate_to_au_feed
# rougher.calculation.floatbank11_sulfate_to_au_feed
# rougher.calculation.sulfate_to_au_concentrate
# rougher.output.concentrate_ag
# rougher.output.concentrate_au
# rougher.output.concentrate_pb
# rougher.output.concentrate_sol
# rougher.output.recovery
# rougher.output.recovery.qa
# rougher.output.tail_ag
# rougher.output.tail_au
# rougher.output.tail_pb
# rougher.output.tail_sol
# secondary_cleaner.output.tail_ag
# secondary_cleaner.output.tail_au
# secondary_cleaner.output.tail_pb
# secondary_cleaner.output.tail_sol

#####################################
# 1.4. Perform data preprocessing.
# (see above) I did this earlier because NaN required attention first.

###################################
# 2.1. Take note of how the concentrations of metals (Au, Ag, Pb) change depending on the purification stage.
# get relevant cols
#[col for col in gold_recovery_full.columns if '_au' in col]
#[col for col in gold_recovery_full.columns if '_ag' in col]
#[col for col in gold_recovery_full.columns if '_pb' in col]

#relevant cols in order of process
#'rougher.input.feed_au' - 'rougher.output.concentrate_au' - 'primary_cleaner.output.concentrate_au' - 'final.output.concentrate_au' 
#'rougher.output.tail_au' - 'primary_cleaner.output.tail_au' - 'secondary_cleaner.output.tail_au' - 'final.output.tail_au'

#'rougher.input.feed_ag' - 'rougher.output.concentrate_ag' - 'primary_cleaner.output.concentrate_ag' - 'final.output.concentrate_ag'
#'rougher.input.feed_ag' - 'rougher.output.tail_ag' - 'primary_cleaner.output.tail_ag' - 'secondary_cleaner.output.tail_ag' - 'final.output.tail_ag'

#'rougher.input.feed_pb' - 'rougher.output.concentrate_pb' - 'primary_cleaner.output.concentrate_pb' - 'final.output.concentrate_pb'
#'rougher.input.feed_pb' - 'rougher.output.tail_pb' - 'primary_cleaner.output.tail_pb' - 'secondary_cleaner.output.tail_pb' - 'final.output.tail_pb' 

# chains ot interate over to plot
chains = {
    "Au Concentrate": [
        'rougher.input.feed_au',
        'rougher.output.concentrate_au',
        'primary_cleaner.output.concentrate_au',
        'final.output.concentrate_au'
    ],
    "Au Tail": [
        'rougher.output.tail_au',
        'primary_cleaner.output.tail_au',
        'secondary_cleaner.output.tail_au',
        'final.output.tail_au'
    ],
    "Ag Concentrate": [
        'rougher.input.feed_ag',
        'rougher.output.concentrate_ag',
        'primary_cleaner.output.concentrate_ag',
        'final.output.concentrate_ag'
    ],
    "Ag Tail": [
        'rougher.input.feed_ag',
        'rougher.output.tail_ag',
        'primary_cleaner.output.tail_ag',
        'secondary_cleaner.output.tail_ag',
        'final.output.tail_ag'
    ],
    "Pb Concentrate": [
        'rougher.input.feed_pb',
        'rougher.output.concentrate_pb',
        'primary_cleaner.output.concentrate_pb',
        'final.output.concentrate_pb'
    ],
    "Pb Tail": [
        'rougher.input.feed_pb',
        'rougher.output.tail_pb',
        'primary_cleaner.output.tail_pb',
        'secondary_cleaner.output.tail_pb',
        'final.output.tail_pb'
    ]
}

# plot
"""fig, axs = plt.subplots(3, 2, figsize=(15, 15))
axs = axs.flatten()  # flatten to iterate over subplots easily

# Iterate over each chain and its corresponding subplot
for idx, (chain_name, steps) in enumerate(chains.items()):
    ax = axs[idx]
    # Compute the average value for each step (requires a DataFrame named 'gold_recovery')
    values = [gold_recovery_full[step].median() for step in steps] # I didnt bother checking if normally distributed to used median.
    
    # Create the bar chart
    bars = ax.bar(steps, values, color='skyblue')
    
    # Set title and labels
    ax.set_title(f"Bar Chart for {chain_name}")
    ax.set_xlabel("Process Step")
    ax.set_ylabel("Average Value")
    
    # Rotate x labels for better readability if needed
    ax.set_xticklabels(steps, rotation=45, ha='right')
    
    # Annotate each bar with its numeric value
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()"""

# extra au vars
#'rougher.calculation.sulfate_to_au_concentrate'
#'rougher.calculation.floatbank10_sulfate_to_au_feed'
#'rougher.calculation.floatbank11_sulfate_to_au_feed'

# extra pb var
#'rougher.calculation.au_pb_ratio'

######################################
# 2.2. Compare the feed particle size distributions in the training set and in the test set. If the distributions vary significantly, the model evaluation will be incorrect.

# I'm just going go compare all vars between training and test.  They should all be similar if they're random subsamples of full df. 
# common columns

#common_columns = set(gold_recovery_train.columns) & set(gold_recovery_test.columns)
#print([col for col in common_columns])


common_columns_sans_date = [
    "rougher.input.feed_ag",
    "rougher.input.feed_au",
    "rougher.input.feed_pb",
    "rougher.input.feed_rate",
    "rougher.input.feed_size",
    "rougher.input.feed_sol",
    "rougher.input.floatbank10_sulfate",
    "rougher.input.floatbank10_xanthate",
    "rougher.input.floatbank11_sulfate",
    "rougher.input.floatbank11_xanthate",
    "rougher.state.floatbank10_a_air",
    "rougher.state.floatbank10_a_level",
    "rougher.state.floatbank10_b_air",
    "rougher.state.floatbank10_b_level",
    "rougher.state.floatbank10_c_air",
    "rougher.state.floatbank10_c_level",
    "rougher.state.floatbank10_d_air",
    "rougher.state.floatbank10_d_level",
    "rougher.state.floatbank10_e_air",
    "rougher.state.floatbank10_e_level",
    "rougher.state.floatbank10_f_air",
    "rougher.state.floatbank10_f_level",
    "primary_cleaner.input.depressant",
    "primary_cleaner.input.feed_size",
    "primary_cleaner.input.sulfate",
    "primary_cleaner.input.xanthate",
    "primary_cleaner.state.floatbank8_a_air",
    "primary_cleaner.state.floatbank8_a_level",
    "primary_cleaner.state.floatbank8_b_air",
    "primary_cleaner.state.floatbank8_b_level",
    "primary_cleaner.state.floatbank8_c_air",
    "primary_cleaner.state.floatbank8_c_level",
    "primary_cleaner.state.floatbank8_d_air",
    "primary_cleaner.state.floatbank8_d_level",
    "secondary_cleaner.state.floatbank2_a_air",
    "secondary_cleaner.state.floatbank2_a_level",
    "secondary_cleaner.state.floatbank2_b_air",
    "secondary_cleaner.state.floatbank2_b_level",
    "secondary_cleaner.state.floatbank3_a_air",
    "secondary_cleaner.state.floatbank3_a_level",
    "secondary_cleaner.state.floatbank3_b_air",
    "secondary_cleaner.state.floatbank3_b_level",
    "secondary_cleaner.state.floatbank4_a_air",
    "secondary_cleaner.state.floatbank4_a_level",
    "secondary_cleaner.state.floatbank4_b_air",
    "secondary_cleaner.state.floatbank4_b_level",
    "secondary_cleaner.state.floatbank5_a_air",
    "secondary_cleaner.state.floatbank5_a_level",
    "secondary_cleaner.state.floatbank5_b_air",
    "secondary_cleaner.state.floatbank5_b_level",
    "secondary_cleaner.state.floatbank6_a_air",
    "secondary_cleaner.state.floatbank6_a_level"
]
"""counter = 0
for col in common_columns_sans_date:
    output = split_test_plot(gold_recovery_train[col],gold_recovery_test[col])
    if abs(output['diff_means']/output['mean_1']) >= 0.10:
        counter += 1
        print(f'{col} diff_means: {round(output["diff_means"]/output["mean_1"],2)}')
print(f'n concerning vars: {counter}') #14"""

##############################################
# 2.3. Consider the total concentrations of all substances at different stages: raw feed, rougher concentrate, and final concentrate. Do you notice any abnormal values in the total distribution? If you do, is it worth removing such values from both samples? Describe the findings and eliminate anomalies. 

# [col for col in gold_recovery_full.columns]

# columns = [
#     "rougher.input.feed_ag",
#     "rougher.input.feed_pb",
#     "rougher.input.feed_rate",
#     "rougher.input.feed_size",
#     "rougher.input.feed_sol",
#     "rougher.input.feed_au",
#     "rougher.output.concentrate_ag",
#     "rougher.output.concentrate_pb",
#     "rougher.output.concentrate_sol",
#     "rougher.output.concentrate_au",
#     "rougher.output.tail_ag",
#     "rougher.output.tail_pb",
#     "rougher.output.tail_sol",
#     "rougher.output.tail_au",
#     "primary_cleaner.output.concentrate_ag",
#     "primary_cleaner.output.concentrate_pb",
#     "primary_cleaner.output.concentrate_sol",
#     "primary_cleaner.output.concentrate_au",
#     "primary_cleaner.output.tail_ag",
#     "primary_cleaner.output.tail_pb",
#     "primary_cleaner.output.tail_sol",
#     "primary_cleaner.output.tail_au",
#     "secondary_cleaner.output.tail_ag",
#     "secondary_cleaner.output.tail_pb",
#     "secondary_cleaner.output.tail_sol",
#     "secondary_cleaner.output.tail_au",
#     "final.output.concentrate_ag",
#     "final.output.concentrate_pb",
#     "final.output.concentrate_sol",
#     "final.output.concentrate_au",
#     "final.output.tail_ag",
#     "final.output.tail_pb",
#     "final.output.tail_sol",
#     "final.output.tail_au",
#     "rougher.calculation.sulfate_to_au_concentrate",
#     "rougher.calculation.au_pb_ratio",
#     "rougher.output.recovery",
#     "final.output.recovery",
#     "primary_cleaner.state.floatbank8_a_air",
#     "primary_cleaner.state.floatbank8_a_level",
#     "primary_cleaner.state.floatbank8_b_air",
#     "primary_cleaner.state.floatbank8_b_level",
#     "primary_cleaner.state.floatbank8_c_air",
#     "primary_cleaner.state.floatbank8_c_level",
#     "primary_cleaner.state.floatbank8_d_air",
#     "primary_cleaner.state.floatbank8_d_level",
#     "rougher.calculation.floatbank10_sulfate_to_au_feed",
#     "rougher.calculation.floatbank11_sulfate_to_au_feed",
#     "rougher.input.floatbank10_sulfate",
#     "rougher.input.floatbank10_xanthate",
#     "rougher.input.floatbank11_sulfate",
#     "rougher.input.floatbank11_xanthate",
#     "rougher.state.floatbank10_a_air",
#     "rougher.state.floatbank10_a_level",
#     "rougher.state.floatbank10_b_air",
#     "rougher.state.floatbank10_b_level",
#     "rougher.state.floatbank10_c_air",
#     "rougher.state.floatbank10_c_level",
#     "rougher.state.floatbank10_d_air",
#     "rougher.state.floatbank10_d_level",
#     "rougher.state.floatbank10_e_air",
#     "rougher.state.floatbank10_e_level",
#     "rougher.state.floatbank10_f_air",
#     "rougher.state.floatbank10_f_level",
#     "secondary_cleaner.state.floatbank2_a_air",
#     "secondary_cleaner.state.floatbank2_a_level",
#     "secondary_cleaner.state.floatbank2_b_air",
#     "secondary_cleaner.state.floatbank2_b_level",
#     "secondary_cleaner.state.floatbank3_a_air",
#     "secondary_cleaner.state.floatbank3_a_level",
#     "secondary_cleaner.state.floatbank3_b_air",
#     "secondary_cleaner.state.floatbank3_b_level",
#     "secondary_cleaner.state.floatbank4_a_air",
#     "secondary_cleaner.state.floatbank4_a_level",
#     "secondary_cleaner.state.floatbank4_b_air",
#     "secondary_cleaner.state.floatbank4_b_level",
#     "secondary_cleaner.state.floatbank5_a_air",
#     "secondary_cleaner.state.floatbank5_a_level",
#     "secondary_cleaner.state.floatbank5_b_air",
#     "secondary_cleaner.state.floatbank5_b_level",
#     "secondary_cleaner.state.floatbank6_a_air",
#     "secondary_cleaner.state.floatbank6_a_level",
#     "primary_cleaner.input.xanthate",
#     "primary_cleaner.input.sulfate",
#     "primary_cleaner.input.depressant",
#     "primary_cleaner.input.feed_size"
# ]

# feed sums
rougher_input_sum_cols = ["rougher.input.feed_ag",
    "rougher.input.feed_pb",
    "rougher.input.feed_sol",
    "rougher.input.feed_au"]


gold_recovery_full['rougher_input_sum'] = gold_recovery_full[rougher_input_sum_cols].sum(axis=1)



# rougher sums
rougher_output_concentrate_sum_cols = ["rougher.output.concentrate_ag",
    "rougher.output.concentrate_pb",
    "rougher.output.concentrate_sol",
    "rougher.output.concentrate_au"]

gold_recovery_full['rougher_output_concentrate_sum'] = gold_recovery_full[rougher_output_concentrate_sum_cols].sum(axis=1)

rougher_output_tail_cols = ["rougher.output.tail_ag",
    "rougher.output.tail_pb",
    "rougher.output.tail_sol",
    "rougher.output.tail_au"]

gold_recovery_full['rougher_output_tail_sum'] = gold_recovery_full[rougher_output_tail_cols].sum(axis=1)

# primary
primary_cleaner_output_sum_cols = ["primary_cleaner.output.concentrate_ag",
    "primary_cleaner.output.concentrate_pb",
    "primary_cleaner.output.concentrate_sol",
    "primary_cleaner.output.concentrate_au"]

gold_recovery_full['primary_cleaner_output_sum'] = gold_recovery_full[primary_cleaner_output_sum_cols].sum(axis = 1)

primary_cleaner_output_tail_sum_cols=[                                                           
    "primary_cleaner.output.tail_ag",
    "primary_cleaner.output.tail_pb",
    "primary_cleaner.output.tail_sol",
    "primary_cleaner.output.tail_au"]

gold_recovery_full['primary_cleaner_output_tail_sum'] = gold_recovery_full[primary_cleaner_output_tail_sum_cols].sum(axis=1)
    
# secondary
secondary_cleaner_output_tail_sum_cols = ["secondary_cleaner.output.tail_ag",
    "secondary_cleaner.output.tail_pb",
    "secondary_cleaner.output.tail_sol",
    "secondary_cleaner.output.tail_au"]
gold_recovery_full['secondary_cleaner_output_tail_sum'] = gold_recovery_full[secondary_cleaner_output_tail_sum_cols].sum(axis=1)
    
# final sums
final_output_concentrate_sum_cols = ["final.output.concentrate_ag",
    "final.output.concentrate_pb",
    "final.output.concentrate_sol",
    "final.output.concentrate_au"]

gold_recovery_full['final_output_concentrate_sum'] = gold_recovery_full[final_output_concentrate_sum_cols].sum(axis = 1)

final_output_tail_sum_cols = [                                                                
    "final.output.tail_ag",
    "final.output.tail_pb",
    "final.output.tail_sol",
    "final.output.tail_au"]

gold_recovery_full['final_output_tail_sum'] = gold_recovery_full[final_output_tail_sum_cols].sum(axis=1)

sum_cols_x = ['rougher_input_sum_cols','rougher_output_concentrate_sum_cols', 'rougher_output_tail_cols', 'primary_cleaner_output_sum_cols', 'primary_cleaner_output_tail_sum_cols', 'secondary_cleaner_output_tail_sum_cols','final_output_concentrate_sum_cols','final_output_tail_sum_cols']
sum_cols_y = ['rougher_input_sum', 'rougher_output_concentrate_sum', 'rougher_output_tail_sum', 'primary_cleaner_output_sum', 'primary_cleaner_output_tail_sum', 'secondary_cleaner_output_tail_sum', 'final_output_concentrate_sum','final_output_tail_sum']

#see(gold_recovery_full[sum_cols_y])

for col in sum_cols_y:
    min = gold_recovery_full[col].min()
    max = gold_recovery_full[col].max()
    print(f'[{min}, {max}]')
    
# notes
# high number of 0s: rougher_output_concentrate_sum, primary_cleaner_output_sum, primary_cleaner_output_tail_sum, secondary_cleaner_output_tail_sum
# noticeable number of 0s: final_output_concentrate_sum, final_output_tail_sum
# 0s might be a missing data label
# I'll drop them, though in reality I'd ask about this. 

##########################################
# add sum cols to dfs and then drop 0 sum rows


# add sum col and drop 0 sums while youre at it
for y, x in zip(sum_cols_y, sum_cols_x):
    if x in gold_recovery_train.columns:
        # add sum col
        gold_recovery_train[y] = gold_recovery_train[globals()[x]].sum(axis = 1)
        # delete 0 sum rows
        gold_recovery_train = gold_recovery_train[(gold_recovery_train[y] != 0).all(axis=1)]
    # repeat for test df
    if x in gold_recovery_test.columns:
        gold_recovery_test[y] = gold_recovery_test[globals()[x]].sum(axis = 1)
        gold_recovery_test = gold_recovery_test[(gold_recovery_test[y] != 0).all(axis=1)]

# test it worked
#(gold_recovery_train[sum_cols_y[0]] == 0).astype('int64').sum() # equals 0
##########################################
#3.1. Write a function to calculate the final sMAPE value.
def smape(
    target:pd.Series = None,
    pred:pd.Series = None
):
    smape = ((target - pred)/ ((target + pred)/2)).mean()*100
    return smape

# Create a scorer that negates SMAPE (so higher is better for sklearn)
cross_val_score_smape = make_scorer(smape, greater_is_better=False)

#########################################
# 3.2. Train different models. Evaluate them using cross-validation. Pick the best model and test it using the test sample. Provide findings.
target = gold_recovery_train['rougher.output.recovery']
features = gold_recovery_train[list(common_columns_sans_date)]
features_test = features.copy()
random_state = 12345

#models to test
linear_regression = LinearRegression()
decision_tree = DecisionTreeRegressor(random_state=random_state)
random_forest = RandomForestRegressor(random_state=random_state)

# data preprocessing for model testing
# feature scaling
scaler = StandardScaler()
features = scaler.fit_transform(features)

#linear regression
# cross-validation
scores = cross_val_score(linear_regression, features, target, cv=5, scoring=cross_val_score_smape)
# un-negate scores back to actual scores
linear_regression_train_rougher_smape = -scores.mean()

print(f'linear_regression_smape: {linear_regression_train_rougher_smape}')
# linear_regression_smape: -4.848766507580829

# decision tree
# cross-validation
scores = cross_val_score(decision_tree, features, target, cv=5, scoring=cross_val_score_smape)
# un-negate scores back to actual scores
decision_tree_train_rougher_smape = -scores.mean()


print(f'decision_tree_smape: {decision_tree_train_rougher_smape}')
# decision_tree_smape: 7.0025241365605755

# random forest
scores = cross_val_score(random_forest, features, target, cv=5, scoring=cross_val_score_smape)
# un-negate scores back to actual scores
random_forest_train_rougher_smape = -scores.mean()

print(f'random_forest_smape: {random_forest_train_rougher_smape}')
# random_forest_smape: 2.421531513432531

# best model: random forest
###################################################
# 3.2 cont.
# second target: 'final.output.recovery'
target = gold_recovery_train['final.output.recovery']
features = gold_recovery_train[list(common_columns_sans_date)]
features_test = features.copy()
random_state = 12345

#models to test
linear_regression = LinearRegression()
decision_tree = DecisionTreeRegressor(random_state=random_state)
random_forest = RandomForestRegressor(random_state=random_state)

# data preprocessing for model testing
# feature scaling
scaler = StandardScaler()
features = scaler.fit_transform(features)

#linear regression
# cross-validation
scores = cross_val_score(linear_regression, features, target, cv=5, scoring=cross_val_score_smape)
# un-negate scores back to actual scores
linear_regression_train_final_smape = -scores.mean()

print(f'linear_regression_smape: {linear_regression_train_final_smape}')
# linear_regression_smape: -0.3471086412168809

# decision tree
# cross-validation
scores = cross_val_score(decision_tree, features, target, cv=5, scoring=cross_val_score_smape)
# un-negate scores back to actual scores
decision_tree_train_final_smape = -scores.mean()


print(f'decision_tree_smape: {decision_tree_train_final_smape}')
# decision_tree_smape: 3.8155620284159495

# random forest
scores = cross_val_score(random_forest, features, target, cv=5, scoring=cross_val_score_smape)
# un-negate scores back to actual scores
random_forest_train_final_smape = -scores.mean()

print(f'random_forest_smape: {random_forest_train_final_smape}')
# random_forest_smape: 1.4514028582927803

# best model: random forest
##############################
# 3.2 cont.
# Final sMAPE
linear_regression_train_final_sMAPE = .25*linear_regression_train_rougher_smape + .75*linear_regression_train_final_smape
decision_tree_train_final_sMAPE = .25*decision_tree_train_rougher_smape + .75*decision_tree_train_final_smape
random_forest_train_final_sMAPE = .25*random_forest_train_rougher_smape + .75*random_forest_train_final_smape

print(f'linear_regression_train_final_sMAPE: {linear_regression_train_final_sMAPE}\
      \ndecision_tree_train_final_sMAPE: {decision_tree_train_final_sMAPE}\
      \nrandom_forest_train_final_sMAPE: {random_forest_train_final_sMAPE}')