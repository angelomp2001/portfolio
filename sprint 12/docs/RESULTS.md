# RESULTS — EXP-005

**Branch:** `experiments/EXP-005-Improve-Readability`  
**Goal:** Refining the project structure by centralizing configuration, standardizing model pipelines, inferring column types, and cleaning up code logic for readability.

## Changes Made
- Changed hardcoded variable arrays to inferred logic (`select_dtypes`).
- Made parameters clear (`k_folds` vs `n_folds`).
- Replaced custom splitting code with standard library `sklearn.model_selection.train_test_split`.
- Enabled the end-of-run `save_best_model` and `evaluate_model` explicitly passing in target array instead of kwargs to `train_test_split`.
- Prepared for classification models code segment in main.

## Model Results (5-fold CV → best model tuned → test evaluation)

**Best model (auto-detected):** CatBoostRegressor  
**Best hyperparameters:** `n_estimators=145`, `max_depth=6`, `learning_rate=0.2`  

| Metric | Test Value |
|--------|-----------|
| RMSE   | 2,094.39  |
| MSE    | 4,386,460.50 |
| MAE    | 1,454.92  |
| R²     | 0.8177    |
| Pred time | 0.0035s |

**Comparison to EXP-004 baseline (RMSE 2,195.02):** improved by ~100 points with standard train-test splitting and refined random generator seeds.
