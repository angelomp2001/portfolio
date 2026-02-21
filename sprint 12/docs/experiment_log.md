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
