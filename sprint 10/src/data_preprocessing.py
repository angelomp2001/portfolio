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
from src.data_explorers import view, see
from src.H0_testing import split_test_plot

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

def load_data(path):
    return pd.read_csv(path)

def data_processing(
        gold_recovery_full,
        gold_recovery_train,
        gold_recovery_test
    ):
    #dropna
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
    # Check that recovery is calculated correctly: rougher.output.recovery feature. 

    # rougher recovery target 
    C = gold_recovery_train['rougher.output.concentrate_au'] # share of gold in the concentrate right after flotation
    F = gold_recovery_train['rougher.input.feed_au'] # share of gold in the feed before flotation
    T = gold_recovery_train['rougher.output.tail_au'] # share of gold in the rougher tails right after flotation
    #print(gold_recovery_train[['rougher.input.feed_au','rougher.output.concentrate_au','rougher.output.tail_au']])

    print(f"means:\n{gold_recovery_train['rougher.input.feed_au'].mean():.02f}\n{gold_recovery_train['rougher.output.concentrate_au'].mean():.02f}\n{gold_recovery_train['rougher.output.tail_au'].mean():.02f}")

    # formula:
    gold_recovery_train['rougher.output.recovery.qa'] = C*(F-T)/F/(C-T)*100

    # MAE
    rougher_output_recovery_MAE = abs(gold_recovery_train['rougher.output.recovery'] - gold_recovery_train['rougher.output.recovery.qa'])

    # marginal difference between the two columns

    ######################################
    # figure
    plt.figure(figsize=(10, 12))

    #First histogram
    plt.subplot(3, 1, 1)
    plt.hist(gold_recovery_train['rougher.output.recovery'].astype('int64'), bins=30, color='blue', alpha=0.7)
    plt.title('Rougher Output Recovery')
    plt.xlabel('Value')
    plt.ylabel('Frequency')


    #Second histogram
    plt.subplot(3, 1, 2)
    plt.hist(gold_recovery_train['rougher.output.recovery.qa'].astype('int64'), bins=30, color='green', alpha=0.7)
    plt.title('Rougher Output Recovery QA')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    #Third histogram (absolute difference)
    plt.subplot(3, 1, 3)
    plt.hist(abs(gold_recovery_train['rougher.output.recovery'].astype('int64') - gold_recovery_train['rougher.output.recovery.qa'].astype('int64')), 
            bins=30, color='red', alpha=0.7)
    plt.title('Absolute Difference Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    #show the plot
    plt.tight_layout()
    plt.show()

    ################################
    # features not available in the test set:
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

    ###################################
    # how the concentrations of metals (Au, Ag, Pb) change depending on the purification stage.

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

    # chains to interate over to plot
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
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.flatten()  # flatten to iterate over subplots easily

    # Iterate over each chain and its corresponding subplot
    for idx, (chain_name, steps) in enumerate(chains.items()):
        ax = axs[idx]

        # Calculate the average value for each step
        values = [gold_recovery_full[step].median() for step in steps] # median is more robust to outliers than mean
        
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
    plt.show()

    # extra au vars
    #'rougher.calculation.sulfate_to_au_concentrate'
    #'rougher.calculation.floatbank10_sulfate_to_au_feed'
    #'rougher.calculation.floatbank11_sulfate_to_au_feed'

    # extra pb var
    #'rougher.calculation.au_pb_ratio'

    return gold_recovery_full, gold_recovery_train, gold_recovery_test

def smape(
    target:pd.Series = None,
    pred:pd.Series = None
):
    return ((target - pred)/ ((target + pred)/2)).mean()*100
    

# a scorer that negates SMAPE (so higher is better for sklearn)
cross_val_score_smape = make_scorer(smape, greater_is_better=False)

def best_model(
    scoring = cross_val_score_smape, 
    gold_recovery_train:pd.DataFrame = None,
    common_columns_sans_date:list = None,
    
):
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
    scores = cross_val_score(linear_regression, features, target, cv=5, scoring=scoring)
    # un-negate scores back to actual scores
    linear_regression_train_rougher_smape = -scores.mean()

    print(f'linear_regression_smape: {linear_regression_train_rougher_smape}')
    # linear_regression_smape: -4.848766507580829

    # decision tree
    # cross-validation
    scores = cross_val_score(decision_tree, features, target, cv=5, scoring=scoring)
    # un-negate scores back to actual scores
    decision_tree_train_rougher_smape = -scores.mean()


    print(f'decision_tree_smape: {decision_tree_train_rougher_smape}')
    # decision_tree_smape: 7.0025241365605755

    # random forest
    scores = cross_val_score(random_forest, features, target, cv=5, scoring=scoring)
    # un-negate scores back to actual scores
    random_forest_train_rougher_smape = -scores.mean()

    print(f'random_forest_smape: {random_forest_train_rougher_smape}')
    # random_forest_smape: 2.421531513432531

    # best model: random forest
    return linear_regression_train_rougher_smape, decision_tree_train_rougher_smape, random_forest_train_rougher_smape

def second_target(
    scoring = cross_val_score_smape,
    gold_recovery_train:pd.DataFrame = None,
    common_columns_sans_date:list = None,
    linear_regression_train_rougher_smape: float= None,
    decision_tree_train_rougher_smape: float= None,
    random_forest_train_rougher_smape: float= None
):
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
    scores = cross_val_score(linear_regression, features, target, cv=5, scoring=scoring)
    # un-negate scores back to actual scores
    linear_regression_train_final_smape = -scores.mean()

    print(f'linear_regression_smape: {linear_regression_train_final_smape}')
    # linear_regression_smape: -0.3471086412168809

    # decision tree
    # cross-validation
    scores = cross_val_score(decision_tree, features, target, cv=5, scoring=scoring)
    # un-negate scores back to actual scores
    decision_tree_train_final_smape = -scores.mean()


    print(f'decision_tree_smape: {decision_tree_train_final_smape}')
    # decision_tree_smape: 3.8155620284159495

    # random forest
    scores = cross_val_score(random_forest, features, target, cv=5, scoring=scoring)
    # un-negate scores back to actual scores
    random_forest_train_final_smape = -scores.mean()

    print(f'random_forest_smape: {random_forest_train_final_smape}')
    # random_forest_smape: 1.4514028582927803

    # best model: random forest
    ##############################
    # Final sMAPE
    linear_regression_train_final_sMAPE = .25*linear_regression_train_rougher_smape + .75*linear_regression_train_final_smape
    decision_tree_train_final_sMAPE = .25*decision_tree_train_rougher_smape + .75*decision_tree_train_final_smape
    random_forest_train_final_sMAPE = .25*random_forest_train_rougher_smape + .75*random_forest_train_final_smape

    print(f'linear_regression_train_final_sMAPE: {linear_regression_train_final_sMAPE}\
        \ndecision_tree_train_final_sMAPE: {decision_tree_train_final_sMAPE}\
        \nrandom_forest_train_final_sMAPE: {random_forest_train_final_sMAPE}')
    
    return 
