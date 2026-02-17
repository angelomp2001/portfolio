
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_predict(df, target_col='product', model=None, test_size=0.25, random_state=42):
    """
    Splits data, trains the model, and generates predictions for the validation set.
    """
    if model is None:
        model = LinearRegression()
        
    features = df.drop(columns=[target_col])
    target = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=test_size, random_state=random_state)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    
    # Return a DataFrame with actual and predicted values for the validation set
    results = pd.DataFrame({
        'actual': y_val,
        'predicted': predictions
    }, index=y_val.index)
    
    score = model.score(X_val, y_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    
    return results, score, rmse

def calculate_profit(target_values, predicted_values, count, revenue_per_unit, budget):
    """
    Calculates profit by selecting the top 'count' wells based on predictions.
    """
    # Select indices of the top 'count' predictions
    probs_sorted = predicted_values.sort_values(ascending=False)
    selected_indices = probs_sorted.head(count).index
    
    # Sum the actual target values for these selected wells
    selected_actuals = target_values.loc[selected_indices]
    total_volume = selected_actuals.sum()
    
    revenue = total_volume * revenue_per_unit
    profit = revenue - budget
    return profit

def bootstrap_profit(results_df, count, revenue_per_unit, budget, repeats=1000, n_samples=500, random_state=42):
    """
    Performs bootstrapping to estimate profit distribution.
    For each iteration, samples 'n_samples' wells, then selects top 'count' best predicted ones to calculate profit.
    """
    state = np.random.RandomState(random_state)
    profits = []
    
    for _ in range(repeats):
        # Match original logic: generate a specific seed for each iteration
        subsample_seed = state.randint(0, 10000)
        
        # Sample with replacement using the generated seed
        subsample = results_df.sample(n=n_samples, replace=True, random_state=subsample_seed)
        
        # Calculate profit for this subsample
        profit_val = calculate_profit(
            target_values=subsample['actual'],
            predicted_values=subsample['predicted'],
            count=count,
            revenue_per_unit=revenue_per_unit,
            budget=budget
        )
        profits.append(profit_val)
        
    return pd.Series(profits)

def analyze_region_profitability(profits):
    """
    Calculates risk and summary statistics from bootstrapped profits.
    """
    mean_profit = profits.mean()
    risk_of_loss = (profits < 0).mean() * 100
    lower = profits.quantile(0.025)
    upper = profits.quantile(0.975)
    
    return {
        'mean_profit': mean_profit,
        'risk_of_loss_percent': risk_of_loss,
        'ci_lower': lower,
        'ci_upper': upper
    }
