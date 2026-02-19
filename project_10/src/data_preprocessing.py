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
import joblib
from src.data_explorers import view, see
from src.H0_testing import split_test_plot

# Hardcoded list of common columns sans date to cleaner main.py
COMMON_COLUMNS_SANS_DATE = [
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


def load_data(path):
    """Loads CSV data from path."""
    return pd.read_csv(path)


def preprocess_data(gold_recovery_full, gold_recovery_train, gold_recovery_test):
    """
    Performs basic data cleaning: type conversion and dropping missing values.
    """
    # Create copies to avoid SettingWithCopy warnings if passed slices
    gold_recovery_full = gold_recovery_full.copy()
    gold_recovery_train = gold_recovery_train.copy()
    gold_recovery_test = gold_recovery_test.copy()

    # Convert date to datetime
    gold_recovery_full['date'] = pd.to_datetime(gold_recovery_full['date'])
    gold_recovery_train['date'] = pd.to_datetime(gold_recovery_train['date'])
    gold_recovery_test['date'] = pd.to_datetime(gold_recovery_test['date'])

    # Drop missing values
    gold_recovery_full = gold_recovery_full.dropna()
    gold_recovery_train = gold_recovery_train.dropna()
    gold_recovery_test = gold_recovery_test.dropna()

    return gold_recovery_full, gold_recovery_train, gold_recovery_test


def verify_recovery_calculation(gold_recovery_train, show_plot=False):
    """
    Verifies that the `rougher.output.recovery` provided in the training set
    is calculated correctly based on the chemical formula.
    Returns the figure object.
    """
    # rougher recovery calculation components
    C = gold_recovery_train['rougher.output.concentrate_au']  # Concentrate
    F = gold_recovery_train['rougher.input.feed_au']          # Feed
    T = gold_recovery_train['rougher.output.tail_au']         # Tail

    print(f"Means:\nFeed: {F.mean():.02f}\nConcentrate: {C.mean():.02f}\nTail: {T.mean():.02f}")

    # Formula calculation
    # Only calculate where F*(C-T) is not 0 to avoid division by zero, though dataset should be clean
    calculated_recovery = C * (F - T) / (F * (C - T)) * 100
    
    # store for comparison (optional modification of dataframe locally)
    gold_recovery_train = gold_recovery_train.copy()
    gold_recovery_train['rougher.output.recovery.qa'] = calculated_recovery

    # MAE Calculation
    mae = abs(gold_recovery_train['rougher.output.recovery'] - gold_recovery_train['rougher.output.recovery.qa']).mean()
    print(f"Mean Absolute Error (MAE) between provided and calculated recovery: {mae:.2e}")

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    
    # 1. Provided
    plt.subplot(3, 1, 1)
    plt.hist(gold_recovery_train['rougher.output.recovery'], bins=30, color='blue', alpha=0.7)
    plt.title('Provided Rougher Output Recovery')
    
    # 2. Calculated
    plt.subplot(3, 1, 2)
    plt.hist(gold_recovery_train['rougher.output.recovery.qa'], bins=30, color='green', alpha=0.7)
    plt.title('Calculated Rougher Output Recovery')

    # 3. Difference
    plt.subplot(3, 1, 3)
    plt.hist(abs(gold_recovery_train['rougher.output.recovery'] - gold_recovery_train['rougher.output.recovery.qa']),
             bins=30, color='red', alpha=0.7)
    plt.title('Absolute Difference')

    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig


def analyze_metal_concentrations(gold_recovery_full, show_plot=False):
    """
    Visualizes how the concentrations of metals (Au, Ag, Pb) change
    across different purification stages.
    Returns the figure object.
    """
    chains = {
        "Au (Gold) Concentrate": [
            'rougher.input.feed_au', 'rougher.output.concentrate_au', 
            'primary_cleaner.output.concentrate_au', 'final.output.concentrate_au'
        ],
        "Au (Gold) Tail": [
            'rougher.output.tail_au', 'primary_cleaner.output.tail_au', 
            'secondary_cleaner.output.tail_au', 'final.output.tail_au'
        ],
        "Ag (Silver) Concentrate": [
            'rougher.input.feed_ag', 'rougher.output.concentrate_ag', 
            'primary_cleaner.output.concentrate_ag', 'final.output.concentrate_ag'
        ],
        "Ag (Silver) Tail": [
            'rougher.input.feed_ag', 'rougher.output.tail_ag', 
            'primary_cleaner.output.tail_ag', 'secondary_cleaner.output.tail_ag', 'final.output.tail_ag'
        ],
        "Pb (Lead) Concentrate": [
            'rougher.input.feed_pb', 'rougher.output.concentrate_pb', 
            'primary_cleaner.output.concentrate_pb', 'final.output.concentrate_pb'
        ],
        "Pb (Lead) Tail": [
            'rougher.input.feed_pb', 'rougher.output.tail_pb', 
            'primary_cleaner.output.tail_pb', 'secondary_cleaner.output.tail_pb', 'final.output.tail_pb'
        ]
    }

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.flatten()

    for idx, (chain_name, steps) in enumerate(chains.items()):
        ax = axs[idx]
        # Use median for robustness
        values = [gold_recovery_full[step].median() for step in steps]
        
        bars = ax.bar(steps, values, color='skyblue')
        ax.set_title(f"Median Concentration: {chain_name}")
        ax.set_ylabel("Concentration")
        ax.set_xticklabels(steps, rotation=45, ha='right')

        # Annotate
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig


def compare_feed_distributions(gold_recovery_train, gold_recovery_test):
    """
    Compares feed particle size distributions between train and test sets.
    """
    # Using the imported list
    common_cols = COMMON_COLUMNS_SANS_DATE
    
    counter = 0
    print("Checking for significant distribution differences...")
    for col in common_cols:
        # Check only feed size columns or key inputs
        if "feed_size" in col:
            output = split_test_plot(gold_recovery_train[col], gold_recovery_test[col])
            # If relative difference in means is > 10%
            if abs(output['diff_means']/output['mean_1']) >= 0.10:
                counter += 1
                print(f'WARNING: {col} differs significantly between train/test (Rel Diff: {output["diff_means"]/output["mean_1"]:.2f})')
    
    if counter == 0:
        print("No significant differences found in feed size distributions.")


def remove_anomalies(gold_recovery_train, gold_recovery_test, gold_recovery_full):
    """
    Calculates total concentrations for each stage and removes rows with 0 sum (anomalies).
    Returns cleaned train/test sets and the list of valid model features.
    """
    # Define summary dictionaries
    sum_cols_config = {
        'rougher_input_sum': ["rougher.input.feed_ag", "rougher.input.feed_pb", "rougher.input.feed_sol", "rougher.input.feed_au"],
        'rougher_output_concentrate_sum': ["rougher.output.concentrate_ag", "rougher.output.concentrate_pb", "rougher.output.concentrate_sol", "rougher.output.concentrate_au"],
        'final_output_concentrate_sum': ["final.output.concentrate_ag", "final.output.concentrate_pb", "final.output.concentrate_sol", "final.output.concentrate_au"],
    }

    # For safety, ensure we are working with copies
    gold_recovery_train = gold_recovery_train.copy()
    gold_recovery_test = gold_recovery_test.copy()

    initial_train_len = len(gold_recovery_train)
    initial_test_len = len(gold_recovery_test)

    # Process Train
    for new_col, components in sum_cols_config.items():
        # Only process if all component columns exist
        if all(c in gold_recovery_train.columns for c in components):
            gold_recovery_train[new_col] = gold_recovery_train[components].sum(axis=1)
            # Filter 0s
            gold_recovery_train = gold_recovery_train[gold_recovery_train[new_col] > 0.01] 

    # Process Test (only using input features available in test)
    input_sum_col = 'rougher_input_sum'
    if all(c in gold_recovery_test.columns for c in sum_cols_config[input_sum_col]):
        gold_recovery_test[input_sum_col] = gold_recovery_test[sum_cols_config[input_sum_col]].sum(axis=1)
        gold_recovery_test = gold_recovery_test[gold_recovery_test[input_sum_col] > 0.01]

    print(f"Rows removed from Train: {initial_train_len - len(gold_recovery_train)}")
    print(f"Rows removed from Test: {initial_test_len - len(gold_recovery_test)}")

    return gold_recovery_train, gold_recovery_test, COMMON_COLUMNS_SANS_DATE


def smape(target, pred):
    """Symmetric Mean Absolute Percentage Error."""
    return ((abs(target - pred) / ((abs(target) + abs(pred)) / 2)).mean()) * 100


def train_and_evaluate_models(gold_recovery_train, common_columns):
    """
    Trains models for both Rougher and Final recovery targets and prints weighted sMAPE.
    """
    # 1. Train for Rougher Recovery
    print("Training models for Rougher Output Recovery...")
    rougher_results = _train_stack(gold_recovery_train, common_columns, 'rougher.output.recovery')
    
    # 2. Train for Final Recovery
    print("Training models for Final Output Recovery...")
    final_results = _train_stack(gold_recovery_train, common_columns, 'final.output.recovery')

    # 3. Calculate Weighted sMAPE
    # Final sMAPE = 25% Rougher + 75% Final
    for model_name in rougher_results:
        final_score = 0.25 * rougher_results[model_name] + 0.75 * final_results[model_name]
        print(f"Final Weighted sMAPE for {model_name}: {final_score:.4f}")

    return


def _train_stack(df, features_list, target_col):
    """
    Helper to train Linear, Decision Tree, and Random Forest for a specific target.
    Returns a dictionary of sMAPE scores.
    """
    target = df[target_col]
    features = df[features_list]
    
    # Scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Custom Scorer
    smape_scorer = make_scorer(smape, greater_is_better=False)
    
    results = {}
    
    # Models
    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(random_state=12345, max_depth=5), 
        'RandomForest': RandomForestRegressor(random_state=12345, n_estimators=50, max_depth=10) 
    }

    for name, model in models.items():
        # CV returns negative values for "greater is better" with error metrics, so we negate back
        scores = cross_val_score(model, features_scaled, target, cv=3, scoring=smape_scorer)
        avg_smape = -scores.mean()
        results[name] = avg_smape
        print(f"  {name} sMAPE: {avg_smape:.4f}")
        
    return results

def train_and_save_best_model(gold_recovery_train, common_columns, output_dir='models'):
    """
    Trains the best performing model (RandomForest) on the full training data
    and saves the model and scaler to disk.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    features = gold_recovery_train[common_columns]
    
    # Create and fit scaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"Scaler saved to {output_dir}/scaler.pkl")
    
    # Targets
    targets = {
        'rougher': 'rougher.output.recovery',
        'final': 'final.output.recovery'
    }
    
    for stage, target_col in targets.items():
        target = gold_recovery_train[target_col]
        
        # Train Best Model (RandomForest)
        model = RandomForestRegressor(random_state=12345, n_estimators=50, max_depth=10)
        model.fit(features_scaled, target)
        
        # Save Model
        save_path = os.path.join(output_dir, f'model_{stage}.pkl')
        joblib.dump(model, save_path)
        print(f"{stage} model saved to {save_path}")
