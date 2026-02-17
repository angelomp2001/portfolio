# Experiment: Split EDA and Analysis (EXP-001)

## Goal
To separate Exploratory Data Analysis (EDA) from the core Analysis logic in `main.py` and generalize helper functions to remove hardcoded dependencies.

## Changes

### 1. New Module: `src/analysis.py`
-   **Created** to encapsulate all modeling and statistical analysis logic.
-   **Functions**:
    -   `train_and_predict`: Handles model training and validation prediction generation. Returns specific metrics (R2, RMSE).
    -   `calculate_profit`: Pure function to calculate profit for a given set of predictions and target values, accepting all economic parameters as arguments.
    -   `bootstrap_profit`: Performs bootstrap resampling to estimate profit distribution.
    -   `analyze_region_profitability`: Computes risk metrics (Risk of Loss, Confidence Intervals) from bootstrapped profits.

### 2. Refactored `src/data_preprocessing.py`
-   **Removed**: `profit`, `stats`, `product_predictions`, `bootstrap_predictions`, `top_200_wells`, and `inputs`.
-   **Retained & Cleaned**: `load_data` (simplified to handle dict/list of paths) and `preprocess_data` (focused on cleaning/deduplication).
-   **Rationale**: This file now strictly handles *data preparation*, adhering to Single Responsibility Principle.

### 3. Refactored `main.py`
-   **Configuration**: Moved all "magic numbers" (Budget, Revenue, Wells count) to top-level constants at the start of the script. This makes the experiment parameters easily tunable.
-   **workflow Split**:
    -   `run_eda(dfs)`: Encapsulates all visualization steps. Can be toggled on/off.
    -   `run_analysis(dfs)`: specific function for the modeling and profit estimation loop.
-   **Logic**: Replaced the implicit global state execution with explicit function calls passing the configuration constants.

## Justification
-   **Modularity**: Splitting EDA and Analysis allows for running the expensive analysis without regenerating plots every time.
-   **Generalization**: Removing hardcoded values (`100_000_000`, `4500`, etc.) from helper functions makes the code reusable for different scenarios or data.
-   **Readability**: Meaningful function names (`calculate_profit` vs `profit`) and explicit arguments make the data flow transparent.
