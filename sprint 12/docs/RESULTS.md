# RESULTS — EXP-002

**Branch:** `experiments/EXP-002-Refactor-Model-Training-Pipeline`  
**Goal:** Refactor the model training pipeline so that the model list, hyperparameter tuning, and test evaluation are all explicit in `main.py`. Replace streaming terminal output with live matplotlib charts. Auto-detect the best model and best hyperparameters at runtime.

## Changes Made
- Replaced monolithic `model_training()` with four focused helpers: `train_candidates`, `tune_model`, `evaluate_model`, `print_summary`
- Model list (`candidates` dict) moved to `main.py` — visible at the top level
- Hyperparameters discovered via `RandomizedSearch`-style loop (20 iterations) — no longer hard-coded
- Best model identified dynamically by lowest validation RMSE — no longer hard-coded
- Live horizontal bar chart replaces per-model print lines during training phase
- Live convergence chart (scatter + best-so-far line) replaces streaming numbers during hyperparameter search
- `_devnull()` context manager + `_SILENCE` dict eliminate all library verbose output
- Charts stay open after script ends via `input()` block in `main.py`
- LinearRegression now properly uses its own OHE validation and test splits
- Installed `lightgbm`, `catboost`, `xgboost` into `[externship]` conda environment

## Model Results (Validation RMSE — baseline comparison, 20-iter random search)

| Model | Validation RMSE | Train Time (s) | Pred Time (s) |
|---|---|---|---|
| LinearRegression | 2,724 | ~0.3 | ~0.01 |
| LGBMRegressor | ~2,063 | ~2.2 | ~0.04 |
| RandomForestRegressor | ~2,177 | ~1.6 | ~0.04 |
| CatBoostRegressor | ~2,026 | ~1.1 | ~0.002 |
| XGBRegressor | ~2,236 | ~14 | ~0.003 |

**Best model (auto-detected):** CatBoostRegressor  
**Best hyperparameters (auto-discovered):** `n_estimators=85  max_depth=7  learning_rate=0.2`  
**Best validation RMSE after tuning:** ~2,233  
**Test RMSE:** 2,232.5048  
