# RESULTS — EXP-015

**Branch:** `experiments/EXP-015-Improve-Readability`  
**Goal:** Remove dependencies on `charts.py` and implement minimalist self-contained plotting variables/functions directly in `data_preprocessing.py` and `model_training.py` while cleaning out `_devnull` and `_SILENCE` logic. 

## Changes Made
- Decoupled visualization helpers. `generate_distribution_figure` uses minimalist standard configurations, and `model_training.py` relies on file-level specific plot rendering configurations. 
- Deleted `src/charts.py`.
- Dropped complex real-time annotation plotting tricks in `model_training` CV output window.
- Dropped output suppression commands `_devnull` and `_SILENCE`. 

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

**Comparison:** No change directly to model performance since this was a pure code refactoring, styling, and pipeline abstraction exercise without modifying mathematical properties.
