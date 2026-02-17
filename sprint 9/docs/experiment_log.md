# Experiment Log

## EXP-002: FastAPI Analysis (2026-02-17)

**Status**: ✅ Success
**Description**: Created a FastAPI web application to interactively upload new geological data, check for duplicates, and visualize analysis results against existing regions.

### Summary of Changes (from Branch_Edit_Summary.md)

#### Goal
Create a FastAPI web application to interactively upload new geological data, check for duplicates, and visualize analysis results against existing regions.

#### Changes
1.  **Docs Reorganization**: Moved `agency_readme.md`, `EXPERIMENTS.md`, `RESULTS.md`, `experiment_log.md` into `docs/`.
2.  **Dependencies**: Added `fastapi`, `uvicorn`, `python-multipart`, `jinja2`, `aiofiles`.
3.  **New Modules**:
    *   `src/utils.py`: Implements `calculate_file_hash` and `check_duplicate` using SHA-256.
    *   `src/visualization.py`: Generates base64-encoded profit distribution (`KDE plot`) and risk (`Bar chart`) plots using `matplotlib`/`seaborn`.
    *   `app.py`: FastAPI implementation serving an HTML interface. Handles file upload, duplicate rejection, analysis orchestration, and report rendering.
4.  **Updated README.md**: Added instructions for running the FastAPI server.

#### Logic Overview
*   **Upload**: User uploads a CSV.
*   **Validation**: Server hashes content, compares with `data/*.csv`. If match found, rejects with exact filename.
*   **Analysis**: Used existing `src/analysis` functions but wrapped in a loop within `app.py` to aggregate results dynamically.
*   **Result**: Returns an HTML page with a summary table sorted by **Profit/Risk Ratio**, and embedded visualization plots.
*   **Recommendation**: Best option is determined quantitatively (Profit/Risk Ratio) and highlighted in the table sort order.


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
