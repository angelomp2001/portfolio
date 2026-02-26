# Experiment Log

Running log of all experiments merged into `main`.  
Updated after each successful merge per the branching workflow in `docs/agent_readme.md`.

---

## EXP-001 — Organize Project Structure ✅

**Branch:** `experiments/EXP-001-Organize-Project-Structure`  
**Date:** 2026-02-20  
**Merged:** 2026-02-20  
**Status:** ✅ Success (organizational — no model changes)

### Code Changes
- Added `README.md` at project root with business case, model results table, project structure, and how-to-run instructions
- Cleaned up `main.py`: removed raw-string comment, added named imports, config block, and section dividers
- Extracted `model_training()` from `src/data_preprocessing.py` into `src/model_training.py`
- Added workflow tracking files: `EXPERIMENTS.md`, `RESULTS.md`, `Branch_Edit_Summary.md`, `experiment_log.md`
- Added `.gitignore` (ignores `catboost_info/`, `__pycache__/`, `.pyc`, `.ipynb_checkpoints/`)

### Results
No model results changed — this was a structural/readability sprint.

---

## EXP-002 — Refactor Model Training Pipeline ✅

**Branch:** `experiments/EXP-002-Refactor-Model-Training-Pipeline`  
**Date:** 2026-02-20  
**Merged:** 2026-02-21  
**Status:** ✅ Success

### Code Changes

**`src/model_training.py` — completely rewritten:**
- Replaced monolithic `model_training()` with four focused helpers: `train_candidates`, `tune_model`, `evaluate_model`, `print_summary`
- `train_candidates(candidates)` — accepts a dict of models, scores each on validation, returns `(results, best_name)`. Displays a live horizontal bar chart that grows as each model finishes; winner highlighted in gold
- `tune_model(...)` — runs a random hyperparameter search (20 iterations) against `PARAM_GRIDS`. Displays a live RMSE convergence chart (scatter + best-so-far line). Skips models with no grid (e.g. `LinearRegression`). Returns `(best_model, best_params)` — nothing hard-coded
- `evaluate_model(model, X_test, y_test)` — applies fitted model to held-out test split, returns `(rmse, pred_time)`
- `print_summary(...)` — prints comparison table with dynamically detected `← best` marker; shows discovered hyperparams and test RMSE
- Added `_SILENCE` dict (per-library verbose suppression kwargs) and `_devnull()` context manager (redirects stdout/stderr during `fit()`/`predict()`) to eliminate all streaming console noise
- Added `PARAM_GRIDS` dict centralising all hyperparameter search spaces
- Added dark-theme chart helpers (`_style_axes`, colour constants)

**`main.py` — completely rewritten:**
- Model list now defined in `main.py` as a `candidates` dict — visible at the top level
- Pipeline is four explicit, readable lines: `train_candidates` → `tune_model` → `evaluate_model` → `print_summary`
- Added `N_ITER_TUNE = 20` config constant for random search iterations
- `LinearRegression` now uses its own OHE validation and test splits (not ML label-encoded splits)
- Added `input()` + `plt.close("all")` at end to keep all chart windows open after script exits

### Results

| Model | Validation RMSE | Train Time (s) | Pred Time (s) |
|---|---|---|---|
| LinearRegression | 2,724 | ~0.3 | ~0.01 |
| LGBMRegressor | ~2,063 | ~2.2 | ~0.04 |
| RandomForestRegressor | ~2,177 | ~1.6 | ~0.04 |
| CatBoostRegressor | ~2,026 | ~1.1 | ~0.002 |
| XGBRegressor | ~2,236 | ~14 | ~0.003 |

**Best model (auto-detected):** CatBoostRegressor  
**Best hyperparameters (auto-discovered):** `n_estimators=85  max_depth=7  learning_rate=0.2`  
**Test RMSE:** 2,232.51  

---

## EXP-013 — Selectable Performance Metric ✅

**Branch:** `experiments/EXP-013-Selectable-Performance-Metric`  
**Date:** 2026-02-21  
**Merged:** 2026-02-21  
**Status:** ✅ Success

### Code Changes

**`src/model_training.py`:**
- Added `mean_absolute_error`, `r2_score` imports from `sklearn.metrics`
- Replaced hardcoded RMSE logic with a `METRIC` module-level variable (fallback default `'rmse'`) and a `_metric_info()` helper that returns `(lower_is_better, label, score_fn)` fresh at each call — so whatever `main.py` sets at runtime is always picked up
- Removed eagerly-computed `_LOWER_IS_BETTER` / `_METRIC_LABEL` constants and `_compute_score()` function
- `train_candidates`, `tune_model`, `evaluate_model`, `print_summary` all call `_metric_info()` at entry — no hardcoded "RMSE" strings remain in the module
- Results dict key renamed `'rmse'` → `'score'`; `evaluate_model` return renamed `rmse` → `score`
- Best-model selection uses `min()` or `max()` depending on direction returned by `_metric_info()`
- All chart titles, axis labels, convergence chart annotations, and print statements use `label` from `_metric_info()` dynamically

**`main.py`:**
- Added `import src.model_training as model_training` to allow runtime propagation
- Added `METRIC` to Config block with option comments:
  ```python
  # Options: 'rmse' – Root Mean Squared Error      (lower is better)
  #          'mse'  – Mean Squared Error            (lower is better)
  #          'mae'  – Mean Absolute Error           (lower is better)
  #          'r2'   – R² coefficient of determination (higher is better)
  METRIC = 'rmse'
  model_training.METRIC = METRIC   # propagate choice to training module
  ```
- `test_rmse` variable renamed `test_score` throughout

### Results

No model results changed — this was a pipeline flexibility experiment (infrastructure only). Default metric remains `'rmse'` so outputs are identical to EXP-002. Changing to `'mae'` or `'r2'` now requires editing exactly one line in `main.py`.

---

## EXP-003 — Add Data Drift Tracking ✅

**Branch:** `experiments/EXP-003-Add-Data-Drift-Tracking`  
**Date:** 2026-02-22  
**Merged:** 2026-02-22  
**Status:** ✅ Success (infrastructure — no model changes)

### Code Changes

**`src/data_preprocessing.py`:**
- Added imports: `os`, `json`, `datetime`
- Added `save_data_stats(df, path, label)`:
  - Computes per-column stats: `dtype`, `null_count`, `null_pct`, `unique_count`
  - Numeric columns: additionally computes `mean`, `std`, `min`, `p25`, `p50`, `p75`, `max`
  - Categorical columns: additionally captures `top_value` (most frequent)
  - Writes a timestamped JSON file to `path`; creates parent dirs if needed

**`main.py`:**
- Added import: `save_data_stats` from `src.data_preprocessing`
- Added `save_data_stats(df, 'data/stats_raw.json', label='raw')` after load/sample — captures raw distribution before any cleaning
- Added `save_data_stats(data['df'], 'data/stats_clean.json', label='clean')` after `preprocess_data()` — captures clean distribution

**`docs/template.md` (new):**
- Project README template pre-filled with business case, folder structure, pipeline overview, and API overview

**`docs/checklist.md` (new):**
- ML best-practices checklist tracking which items are implemented (✅) vs outstanding ([ ])

### Results
No model results changed — infrastructure/tooling experiment only.  
**Test RMSE (unchanged):** 2,232.51

---

## EXP-004 — Complete Checklist ✅

**Branch:** `experiments/EXP-004-Complete-Checklist`  
**Date:** 2026-02-22  
**Merged:** 2026-02-23  
**Status:** ✅ Success

### Code Changes

**`src/charts.py` (new):**
- Shared dark-theme constants and helpers (`style_axes`, `new_figure`) imported by both modules

**`src/data_preprocessing.py` (rewrite):**
- Explicit column-role constants: `TARGET_COL`, `NUMERIC_COLS`, `CATEGORICAL_COLS`
- `clean_data(df)` — cleaning only; encoding/scaling moved into Pipelines
- `split_data(df)` — 80/20 holdout split
- `build_preprocessor(cat_cols, num_cols, is_tree)` — ColumnTransformer; non-tree: OHE + PolynomialFeatures(2) on numeric; tree: OrdinalEncoder + passthrough
- `visualize_data(df, label, out_path)` — histograms + bar charts, reused for raw and clean

**`src/model_training.py` (rewrite):**
- `build_keras_model()` — Dense(128→64→32→1) with Dropout, Adam, MSE loss
- `KerasRegressorWrapper` — sklearn-compatible; EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- `train_candidates()` — KFold(k=5) CV; tracks train time + peak memory via tracemalloc; live bar chart ±std
- `plot_fold_scores()` — per-model fold-score line chart (x=fold)
- `plot_keras_history()` — epoch vs train/val loss
- `evaluate_model()` — RMSE, MSE, MAE, R², pred_time
- `save_best_model()` — joblib for sklearn, .keras for NN; writes metadata JSON

**`main.py` (rewrite):**
- Visualize raw + clean data; build preprocessors; 6 pipelines; 5-fold CV → tune → evaluate → save

### Results

**Best model:** CatBoostRegressor  
**Best params:** `n_estimators=140  max_depth=7  learning_rate=0.1`

| Metric | Value |
|--------|-------|
| RMSE   | 2,195.02 |
| MSE    | 4,818,120.58 |
| MAE    | 1,562.50 |
| R²     | 0.7998 |

vs EXP-002 baseline RMSE 2,232.51 — improved by ~37 points.

---

## EXP-005 — Improve Readability ✅

**Branch:** `experiments/EXP-005-Improve-Readability`  
**Date:** 2026-02-23  
**Merged:** 2026-02-23  
**Status:** ✅ Success

### Code Changes
- Refined project structure by centralizing configuration, standardizing model pipelines, and improving code organization.
- Inferred column data types dynamically instead of hardcoding them.
- Renamed variables for clarity (e.g., `n_folds` to `k_folds`).
- Replaced custom data splitting functions with native `train_test_split(X, y)`.
- Prepared for classification tasks by adding a commented-out classification model list in `main.py`.
- Uncommented `evaluate_model` and `save_best_model` to save metrics.

### Results

**Best model:** CatBoostRegressor  
**Best hyperparameters:** `n_estimators=145`, `max_depth=6`, `learning_rate=0.2`

| Metric | Value |
|--------|-------|
| RMSE   | 2,094.39 |
| MSE    | 4,386,460.50 |
| MAE    | 1,454.92 |
| R²     | 0.8177 |

vs EXP-004 baseline RMSE 2,195.02 — improved by ~100 points.

---

## EXP-014 — Refactor Preprocessor ✅

**Branch:** `experiments/EXP-014-Refactor-Preprocessor`  
**Date:** 2026-02-26  
**Merged:** 2026-02-26  
**Status:** ✅ Success

### Code Changes
- Inlined the `num_pipeline` variable natively into the `ColumnTransformer`.
- Added developer inline comments to detail step-by-step logic during preprocessing.
- Triggered metrics logging to properly save results in the latest format.

### Results

**Best model:** CatBoostRegressor  
**Best hyperparameters:** `n_estimators=145`, `max_depth=6`, `learning_rate=0.2`  

| Metric | Test Value |
|--------|-----------|
| RMSE   | 2,094.39  |
| MSE    | 4,386,460.50 |
| MAE    | 1,454.92  |
| R²     | 0.8177    |
| Pred time | ~0.004s |

**Comparison to EXP-005 baseline (RMSE 2,094.39):** No change since this was a pure refactoring and documentation exercise.

---
