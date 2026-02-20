# Branch Edit Summary — EXP-001-Organize-Project-Structure

## Summary
This branch contains only organizational and readability changes. No model logic, hyperparameters, or data pipeline steps were modified.

---

## Files Created

### `README.md`
- New file at project root
- Contains: business case, model comparison table, project structure, how-to-run, data pipeline overview, dependency list

### `src/model_training.py`
- Extracted `model_training()` function from `src/data_preprocessing.py`
- Added consistent `[ModelName]` prefix to all print statements
- Added final summary comparison table
- Final test-set evaluation now uses tuned CatBoost (`n_estimators=87, max_depth=4`) for a clean result
- Imports are scoped to only what this module needs

### `EXPERIMENTS.md`
- New tracking file (required by `docs/agent_readme.md` workflow)
- Records experiment ID, branch, description, status, and RMSE

### `RESULTS.md`
- New tracking file (required by `docs/agent_readme.md` workflow)
- Records results for the current branch

### `Branch_Edit_Summary.md`
- This file

### `experiment_log.md`
- New running log (required by `docs/agent_readme.md` workflow)
- To be updated on `main` after merge

### `.gitignore`
- Added ignore rules for: `catboost_info/`, `__pycache__/`, `.pyc`, `.ipynb_checkpoints/`, `.DS_Store`

---

## Files Modified

### `main.py`
- **Removed:** Large raw-string comment block (lines 26–54) — content moved to `README.md`
- **Changed:** `from src.data_preprocessing import *` → explicit named imports: `load_data`, `preprocess_data`
- **Added:** `from src.model_training import model_training` (now a separate module)
- **Added:** Config block at top (`DATA_PATH`, `SAMPLE_SIZE`, `RANDOM_STATE`)
- **Added:** Section divider comments (`# ── Load`, `# ── Preprocess`, `# ── Train & Evaluate`)
- **Reformatted:** Multi-line tuple unpacking from `preprocess_data()` for readability

### `src/data_preprocessing.py`
- **Removed:** `model_training()` function and its associated imports (`xgboost`, `lightgbm`, `catboost`, `LinearRegression`, `RandomForestRegressor`, `time`)
- No logic changes to `preprocess_data()`
