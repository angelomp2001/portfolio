'''
'''
from src.data_preprocessing import *
from src.data_explorers import view, see





# data
gold_recover_full_path = 'data/gold_recovery_full.csv'
gold_recovery_train_path = 'data/gold_recovery_train.csv'
gold_recovery_test_path = 'data/gold_recovery_test.csv'

gold_recovery_full = load_data(gold_recover_full_path)
gold_recovery_train = load_data(gold_recovery_train_path)
gold_recovery_test = load_data(gold_recovery_test_path)

##############################################
# a closer look

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

'''
df_full notes
no duplicate values
all cols seem relevant
all cols are continuous
date col should be dtype datetime
target has the most missing: rougher.output.recovery: 3119 (14%), but we have lots of data, so dropna() it is. 

df_train notes
date col should be dtype datetime
target has the most missing: rougher.output.recovery: 2573 (15%), but we have lots of data, so dropna() it is. 

df_test notes
date col should be dtype datetime
target is missing: rougher.output.recovery: predict using features.
'''

# max rows (the index column)
# print(max([views['headers'][row]['Unique Values'] for row in range(len(views['headers']))])) 

#############################
# data processing
gold_recovery_full, gold_recovery_train, gold_recovery_test = data_processing(
    gold_recovery_full,
    gold_recovery_train,
    gold_recovery_test
)

######################################
# feed particle size distributions et al in the training set vs in the test set. They need to be similar, otherwise the model will not work well.

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

'''
counter = 0
for col in common_columns_sans_date:
    output = split_test_plot(gold_recovery_train[col],gold_recovery_test[col])
    if abs(output['diff_means']/output['mean_1']) >= 0.10:
        counter += 1
        print(f'{col} diff_means: {round(output["diff_means"]/output["mean_1"],2)}')
print(f'n concerning vars: {counter}') #14
'''

##############################################
#  total concentrations of all substances at different stages: raw feed, rougher concentrate, and final concentrate. checking for abnormal values. 

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
# I'll drop them, though this should be confirmed by the client. 

##########################################
# add sum col and drop 0 sums
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

#########################################
# Training different models. Evaluating using cross-validation. Pick the best model.
linear_regression_train_rougher_smape, decision_tree_train_rougher_smape, random_forest_train_rougher_smape = best_model(
    gold_recovery_train=gold_recovery_train,
    common_columns_sans_date=common_columns_sans_date)
###################################################
# second target: 'final.output.recovery'
second_target(
    gold_recovery_train=gold_recovery_train,
    common_columns_sans_date=common_columns_sans_date,
    linear_regression_train_rougher_smape=linear_regression_train_rougher_smape,
    decision_tree_train_rougher_smape=decision_tree_train_rougher_smape,
    random_forest_train_rougher_smape=random_forest_train_rougher_smape
)

