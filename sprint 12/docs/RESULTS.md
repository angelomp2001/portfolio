# RESULTS — EXP-014

**Branch:** `experiments/EXP-014-Refactor-Preprocessor`  
**Goal:** Inline the numerical preprocessing pipeline and add inline documentation comments.

## Changes Made
- Inlined the `num_pipeline` variable natively into the `ColumnTransformer`.
- Added developer inline comments to detail step-by-step logic during preprocessing.

## Model Results (5-fold CV → best model tuned → test evaluation)

**Best model:** CatBoostRegressor  
**Best hyperparameters:** `n_estimators=145`, `max_depth=6`, `learning_rate=0.2`  

| Metric | Test Value |
|--------|-----------|
| RMSE   | 2,094.39  |
| MSE    | 4,386,460.50 |
| MAE    | 1,454.92  |
| R²     | 0.8177    |
| Pred time | ~0.004s |

**Comparison:** No change directly to model performance since this was a pure code refactoring and pipeline abstraction exercise without modifying mathematical properties.
