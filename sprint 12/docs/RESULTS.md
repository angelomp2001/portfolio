# RESULTS — EXP-001

**Branch:** `experiments/EXP-001-Organize-Project-Structure`  
**Goal:** Organize project folder for clarity and maintainability. No model changes.

## Changes Made
- Added `README.md` at project root
- Cleaned up `main.py` (removed raw-string comment, named imports, config block, section dividers)
- Extracted `model_training()` from `data_preprocessing.py` into `src/model_training.py`
- Added workflow tracking files: `EXPERIMENTS.md`, `RESULTS.md`, `Branch_Edit_Summary.md`, `experiment_log.md`
- Added `.gitignore`

## Model Results
No model training was changed. Results are unchanged from baseline:

| Model | RMSE | Train Time (s) | Pred Time (s) |
|---|---|---|---|
| LinearRegression | 2724.06 | 0.26 | 0.025 |
| LGBMRegressor | 2063.19 | 2.22 | 0.041 |
| RandomForestRegressor | 2176.82 | 1.59 | 0.040 |
| CatBoostRegressor | 2026.19 | 1.09 | 0.002 |
| XGBRegressor | 2236.42 | 14.43 | 0.003 |

CatBoost tuned (max_depth=4): **RMSE = 2008.14**
