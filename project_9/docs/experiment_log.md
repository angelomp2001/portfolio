# Experiment Log

## EXP-001: Split EDA and Analysis (2026-02-17)

**Status**: ✅ Success
**Description**: Split EDA from Analysis, Generalize Functions

### Summary of Changes (from Branch_Edit_Summary.md)
#### Goal
To separate Exploratory Data Analysis (EDA) from the core Analysis logic in `main.py` and generalize helper functions to remove hardcoded dependencies.

#### Changes
1.  **New Module: `src/analysis.py`**
    -   Encapsulates all modeling and statistical analysis logic (`train_and_predict`, `calculate_profit`, `bootstrap_profit`, `analyze_region_profitability`).
2.  **Refactored `src/data_preprocessing.py`**
    -   Removed analysis logic, retained only data loading and cleaning.
3.  **Refactored `main.py`**
    -   Configuration constants moved to top.
    -   Workflow split into `run_eda` and `run_analysis`.
    -   Explicit function calls replacing global state.

#### Justification
-   **Modularity**: Allows running analysis without regenerating plots.
-   **Generalization**: Removes hardcoded values for reusability.
-   **Readability**: Meaningful names and explicit arguments.

### Results (from RESULTS.md)
**Parameters**: Budget=$100_000_000, Revenue/Unit=$4500, Wells=200, Samples=1000

#### Regional Performance Summary
| region   |   mean_profit |   risk_of_loss_percent |         ci_lower |    ci_upper |
|:---------|--------------:|-----------------------:|-----------------:|------------:|
| region_1 |   6.19885e+06 |                    1.5 | 449910           | 1.23181e+07 |
| region_2 |   6.4505e+06  |                    0.6 |      1.47326e+06 | 1.18792e+07 |
| region_3 |   5.77572e+06 |                    2.8 | -77028.1         | 1.23498e+07 |

**Recommended Region**: region_2

---
