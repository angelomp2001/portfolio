# RESULTS — EXP-004

**Branch:** `experiments/EXP-004-Complete-Checklist`  
**Goal:** Complete the project checklist: add Keras NN, 5-fold CV, sklearn Pipelines, PolynomialFeatures for non-tree models, data visualizations, multi-metric evaluation, peak memory tracking, and model saving.

## Changes Made
- Added `src/charts.py` — shared dark-theme chart helpers (reused by both modules)
- Rewrote `src/data_preprocessing.py` — `clean_data()`, `split_data()`, `build_preprocessor()` (Pipeline-ready), `visualize_data()`
- Rewrote `src/model_training.py` — `KerasRegressorWrapper`, `build_keras_model()`, 5-fold CV in `train_candidates()`, `plot_fold_scores()`, `plot_keras_history()`, multi-metric `evaluate_model()`, `save_best_model()`
- Rewrote `main.py` — new pipeline-based flow with visualizations and model saving

## Model Results (5-fold CV → best model tuned → test evaluation)

**Best model (auto-detected):** CatBoostRegressor  
**Best hyperparameters:** `n_estimators=140  max_depth=7  learning_rate=0.1`  

| Metric | Test Value |
|--------|-----------|
| RMSE   | 2,195.02  |
| MSE    | 4,818,120.58 |
| MAE    | 1,562.50  |
| R²     | 0.7998    |
| Pred time | 0.0048s |

**Comparison to EXP-002 baseline (RMSE 2,232.51):** improved by ~37 points with the new CV + Pipeline architecture.
