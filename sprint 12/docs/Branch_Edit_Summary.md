# Branch Edit Summary — EXP-002-Refactor-Model-Training-Pipeline

## Summary
Complete refactor of the model training pipeline for transparency, automation, and a cleaner UX. The monolithic `model_training()` function was replaced with four focused, reusable helpers. The model list, tuning step, and test evaluation are now all explicit, separate lines in `main.py`. Hyperparameter search no longer hard-codes best values — it discovers them automatically via random search. Terminal streaming numbers were replaced with live-updating matplotlib charts.

---

## Files Modified

### `src/model_training.py`
**Completely rewritten.** The old `model_training()` monolith was removed and replaced with four focused functions:

- **`train_candidates(candidates)`**
  - Accepts a dict of `{name: {model, X_train, y_train, X_valid, y_valid}}` — models no longer hard-coded inside the function
  - Returns `(results_dict, best_name)` — best model is discovered at runtime, not hard-coded
  - Displays a **live horizontal bar chart** (dark theme) that grows as each model finishes; winner highlighted in gold

- **`tune_model(model, X_train, y_train, X_valid, y_valid, n_iter, random_state)`**
  - Runs a **manual random search** over `PARAM_GRIDS[model_class_name]`
  - Displays a **live convergence chart** (scatter + best-so-far line) updating every iteration
  - Automatically skips models with no entry or empty grid (e.g. `LinearRegression`)
  - Re-fits the winning configuration on the full training set before returning
  - Returns `(best_model, best_params)` — params are discovered, not hard-coded

- **`evaluate_model(model, X_test, y_test)`**
  - Single-responsibility: applies a fitted model to the held-out test split
  - Returns `(rmse, pred_time)`

- **`print_summary(results, best_name, best_params, test_rmse, test_pred_time)`**
  - Prints the comparison table with `← best` marker on the dynamically detected winner
  - Prints discovered best hyperparameters and final test RMSE — nothing hard-coded

**New module-level additions:**
- `_SILENCE` dict: per-library verbose/verbosity suppression kwargs
- `_devnull()` context manager: redirects `sys.stdout`/`sys.stderr` to `/dev/null` around every `fit()`/`predict()` call, eliminating streaming console noise
- `PARAM_GRIDS` dict: hyperparameter search spaces for each model class (centralised, not buried in loops)
- Dark-theme chart helpers (`_BG`, `_PANEL`, `_TEXT`, `_MUTED`, `_BORDER`, `_COLORS`, `_style_axes()`)
- `matplotlib.use("TkAgg")` for interactive chart rendering

### `main.py`
**Completely rewritten for top-level transparency.**

- **Added imports:** `matplotlib.pyplot`, all five model classes (`LinearRegression`, `LGBMRegressor`, `RandomForestRegressor`, `CatBoostRegressor`, `XGBRegressor`), and the four new helpers from `model_training`
- **Added config constant:** `N_ITER_TUNE = 20` (random search iterations)
- **Added `candidates` dict:** explicit definition of all five models with their correct data splits (OHE for LinearRegression, label-encoded for tree models) — model list is now visible in `main.py`
- **Separated pipeline into four explicit lines:**
  ```python
  results, best_name = train_candidates(candidates)
  best_model, best_params = tune_model(...)
  test_rmse, test_pred_time = evaluate_model(best_model, ...)
  print_summary(results, best_name, best_params, test_rmse, test_pred_time)
  ```
- **Added at end:** `input("Press Enter to close charts…")` + `plt.close("all")` keeps all chart windows open after script completes

### `README.md`
- Updated "Best model" line to note that the winner is now auto-detected
- Updated "How to Run" to mention the live charts and Enter-to-exit prompt

---

## What Was Removed / Fixed
| Before | After |
|---|---|
| `model_training()` monolith — model list hidden inside function | Model list defined in `main.py` as `candidates` dict |
| `n_estimators=87, max_depth=4` hard-coded | Discovered by random search at runtime |
| `← best` hard-coded on CatBoost row | Applied to whichever model wins by lowest RMSE |
| 50+ lines of streaming numbers during hyperparameter search | Live convergence chart |
| Each model's training printed line-by-line | Live bar chart that grows model by model |
| LightGBM/CatBoost/XGBoost printing their own verbose logs | Silenced via `_SILENCE` dict + `_devnull()` context |
| Test data evaluated only on CatBoost, hard-coded | Evaluated on whichever model wins, using its correct data split |
