'''
Find the best place to look for oil, accounting for profit and risk of loss.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import datetime

# Custom modules
from src.data_explorers import view, see
from src.data_preprocessing import load_data, preprocess_data
from src.analysis import train_and_predict, bootstrap_profit, analyze_region_profitability

# ==========================================
# Configuration / Constants
# ==========================================
EXPERIMENT_ID = "EXP-001"
EXPERIMENT_DESC = "Split EDA from Analysis, Generalize Functions"
BUDGET = 100_000_000  # $100 million
WELLS_TO_SELECT = 200  # Top 200 wells to develop
REVENUE_PER_BARREL = 4.5  # Revenue per barrel
REVENUE_PER_UNIT = 4500  # Revenue per unit (1000 barrels)
POINTS_STUDIED = 500  # Number of points to sample for bootstrap
BOOTSTRAP_SAMPLES = 1000  # Number of bootstrap iterations
RANDOM_STATE = 42

DATA_PATHS = {
    'region_1': 'data/geo_data_0.csv',
    'region_2': 'data/geo_data_1.csv',
    'region_3': 'data/geo_data_2.csv'
}

def run_eda(dfs):
    """
    Executes Exploratory Data Analysis steps.
    """
    print("\n--- Starting Exploratory Data Analysis ---\n")
    for name, df in dfs.items():
        print(f"Inspecting {name}:")
        view(df, view='headers')
        see(df, x=name)
        print("-" * 30)
    print("\n--- EDA Complete ---\n")

def run_analysis(dfs):
    """
    Executes the core analysis: model training, profit calculation, and risk assessment.
    """
    print("\n--- Starting Profitability Analysis ---\n")
    
    results_summary = []
    
    for region, df in dfs.items():
        print(f"Analyzing {region}...")
        
        # Train and Evaluate Model
        model = LinearRegression()
        predictions_df, score, rmse = train_and_predict(
            df, 
            target_col='product', 
            model=model, 
            test_size=0.25, 
            random_state=RANDOM_STATE
        )
        
        print(f"  Model Score (R2): {score:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        # Bootstrap Profit Distribution
        profits = bootstrap_profit(
            results_df=predictions_df,
            count=WELLS_TO_SELECT,
            revenue_per_unit=REVENUE_PER_UNIT,
            budget=BUDGET,
            repeats=BOOTSTRAP_SAMPLES,
            n_samples=POINTS_STUDIED,
            random_state=RANDOM_STATE
        )
        
        # Calculate Risk and Statistics
        stats = analyze_region_profitability(profits)
        stats['region'] = region
        results_summary.append(stats)
        
        print(f"  Mean Profit: ${stats['mean_profit']:,.2f}")
        print(f"  Risk of Loss: {stats['risk_of_loss_percent']:.2f}%")
        print(f"  95% Confidence Interval: (${stats['ci_lower']:,.2f}, ${stats['ci_upper']:,.2f})")
        print("-" * 30)

    # Final Recommendation
    print("\n--- Final Recommendation ---\n")
    summary_df = pd.DataFrame(results_summary).set_index('region')
    print(summary_df)
    
    best_region = summary_df[summary_df['risk_of_loss_percent'] < 2.5]['mean_profit'].idxmax()
    print(f"\nRecommended Region for Development: {best_region}")
    
    return summary_df, best_region

def log_results(summary_df, best_region):
    """
    Logs the experiment results to RESULTS.md and EXPERIMENTS.md.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the summary table for markdown
    markdown_table = summary_df.to_markdown()
    
    # 1. Log to RESULTS.md
    with open("docs/RESULTS.md", "a") as f:
        f.write(f"\n## Results for {EXPERIMENT_ID} - {timestamp}\n")
        f.write(f"**Description**: {EXPERIMENT_DESC}\n")
        f.write(f"**Parameters**: Budget=${BUDGET:_}, Revenue/Unit=${REVENUE_PER_UNIT}, Wells={WELLS_TO_SELECT}, Samples={BOOTSTRAP_SAMPLES}\n\n")
        f.write(f"### Regional Performance Summary\n")
        f.write(markdown_table + "\n\n")
        f.write(f"**Recommended Region**: {best_region}\n")
        f.write("-" * 40 + "\n")
        
    print(f"Results logged to RESULTS.md")

    # 2. Log to EXPERIMENTS.md (High-level log)
    with open("docs/EXPERIMENTS.md", "a") as f:
        f.write(f"| {timestamp} | {EXPERIMENT_ID} | {EXPERIMENT_DESC} | {best_region} | Success |\n")
        
    print(f"Experiment logged to EXPERIMENTS.md")

def main():
    # 1. Load Data
    raw_dfs = load_data(DATA_PATHS)
    
    # 2. Preprocess
    dfs = preprocess_data(raw_dfs)
    
    # 3. Exploratory Data Analysis (Optional / Configurable)
    # Uncomment next line to run EDA
    # run_eda(dfs)
    
    # 4. Analysis
    summary_df, best_region = run_analysis(dfs)
    
    # 5. Log Results
    log_results(summary_df, best_region)

if __name__ == "__main__":
    main()