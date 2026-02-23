# Branch Edit Summary — EXP-005-Improve-Readability

## Summary
Refined project structure by centralizing configuration, standardizing model pipelines, and improving code organization. Inferred column data types dynamically instead of hardcoding them. Renamed variables for clarity (e.g., `n_folds` to `k_folds`), replaced custom data splitting functions with standard library equivalents (`train_test_split`), and prepared for classification tasks by adding a commented-out classification model list in `main.py`.

---

## Files Modified

### `main.py`
- Imported `train_test_split` from `sklearn.model_selection`.
- Renamed parameter `n_folds` to `k_folds` for consistency.
- Standardized the splitting to correctly use standard positional arguments with `train_test_split(X, y)`.
- Added a `MODELS_CLASSIFICATION` array that is commented out, allowing for an easy switch to classification tasks in the future.
- Removed hardcoded lists for numeric and categorical columns, replacing them with dynamic dtype inference (`select_dtypes`).
- Uncommented `evaluate_model` and `save_best_model` to save actual pipeline metrics and predictions at the end of the run.

### `src/data_preprocessing.py`
- Dropped the hardcoded column lists (`CATEGORICAL_COLS`, `NUMERIC_COLS`).
- Relying entirely on dynamically selecting numeric or categorical columns based on their inferred dtype.
- Replaced custom 80-20 split implementation with `scikit-learn`'s `train_test_split` (now handled in `main.py`).

### `src/model_training.py`
- Parameter renames for clarity throughout function signatures (e.g. `k_folds` instead of `n_folds` in `train_candidates`). 
- Minor comment wording updates to reflect standardization.
