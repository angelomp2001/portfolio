# Branch Edit Summary — EXP-004-Complete-Checklist

## Summary
Major pipeline refactor to complete the project checklist. Introduced sklearn Pipelines with separate preprocessors for tree vs non-tree models, 5-fold CV replacing the single 60/20/20 split, PolynomialFeatures on numeric columns for linear/NN models, a Keras NN with Dropout/EarlyStopping/ReduceLROnPlateau/ModelCheckpoint, multi-metric evaluation (RMSE/MSE/MAE/R²), data + training visualizations, peak memory tracking, and model saving.

---

## Files Added

### `src/charts.py` (new)
- Shared dark-theme chart helpers: `BG, PANEL, TEXT, MUTED, BORDER, COLORS`, `style_axes()`, `new_figure()`
- Imported by both `data_preprocessing.py` and `model_training.py` to reuse styling

---

## Files Modified

### `src/data_preprocessing.py` (rewrite)
- **Added imports:** `tracemalloc`, `matplotlib`, `sklearn.pipeline.Pipeline`, `sklearn.compose.ColumnTransformer`, `PolynomialFeatures`, `OrdinalEncoder`, `src.charts`
- **Added column-role constants:** `TARGET_COL`, `NUMERIC_COLS`, `CATEGORICAL_COLS`, `COLS_TO_DROP` — columns now explicitly labeled by type (✅ Label numeric/categorical/target)
- **Extracted `clean_data(df)`:** cleaning-only step (drop, fix ranges, fill NaN, dedup, dropna); no encoding/scaling
- **Added `split_data(df)`:** 80/20 holdout split returning `X_train, X_test, y_train, y_test`
- **Added `build_preprocessor(cat_cols, num_cols, is_tree)`:**
  - `is_tree=False` (linear/NN): OHE for categorical + PolyFeatures(degree=2)+StandardScaler for numeric (✅ PolynomialFeatures)
  - `is_tree=True` (tree models): OrdinalEncoder for categorical + passthrough for numeric
- **Added `visualize_data(df, label, out_path)`:** histograms for numeric cols, bar charts (top-10) for categorical; saves PNG; reusable for raw and clean (✅ Visualize raw/clean data)
- **Kept `save_data_stats()`** unchanged

### `src/model_training.py` (rewrite)
- **Replaced chart helpers** with imports from `src.charts`
- **Added:** `import tracemalloc`, `import joblib`, `from sklearn.base import clone`, `from sklearn.model_selection import KFold`, `from sklearn.metrics import mean_absolute_error, r2_score`, `import tensorflow`, `import keras`
- **Added `build_keras_model(input_dim, dropout_rate, learning_rate)`:** Dense(128)→Dropout→Dense(64)→Dropout→Dense(32)→Dense(1), compiled with Adam (✅ Keras NN, ✅ Dropout, ✅ Learning rate)
- **Added `KerasRegressorWrapper`:** sklearn-compatible wrapper with `fit/predict/get_params/set_params`; callbacks: EarlyStopping(patience=15) ✅, ReduceLROnPlateau(factor=0.5) ✅, ModelCheckpoint(save_best_only=True, final fit only) ✅
- **Rewrote `train_candidates(pipelines, X_train, y_train)`:**
  - KFold(k=5, shuffle=True) CV (✅ Cross validation)
  - `clone(pipeline)` per fold for leakage-safe refit
  - Tracks train time + peak memory via `tracemalloc` per model (✅ Training statistics)
  - Live comparison bar chart with ±std error bars
  - Calls `plot_fold_scores()` after all models finish
- **Added `plot_fold_scores(results)`:** per-model subplot, x=fold, y=RMSE (✅ Training visualization timeseries)
- **Added `plot_keras_history(history, name)`:** epoch vs train/val loss chart (✅ Training visualization timeseries)
- **Rewrote `tune_model(pipeline, model_name, X_train, y_train, ...)`:** uses `pipeline.set_params(model__k=v)` to tune via Pipeline; uses fixed 80/20 validation split of train pool; live convergence chart kept
- **Rewrote `evaluate_model(pipeline, X_test, y_test)`:** returns dict with `rmse, mse, mae, r2, pred_time` (✅ Evaluation metrics)
- **Added `save_best_model(pipeline, name, params, metrics, out_dir)`:** joblib for sklearn pipelines, `.keras` + `.joblib` preprocessor for Keras; writes `best_model_metadata.json` (✅ Save best model)
- **Rewrote `print_summary()`:** shows CV RMSE ± std, train time, peak memory; shows test RMSE/MSE/MAE/R²

### `main.py` (rewrite)
- **New imports:** `Pipeline`, all new helpers from `data_preprocessing` and `model_training`
- **New flow:**
  1. Load + sample → `save_data_stats` (raw) + `visualize_data` (raw)
  2. `clean_data` → `save_data_stats` (clean) + `visualize_data` (clean)
  3. `split_data` → `X_train, X_test, y_train, y_test`
  4. `build_preprocessor` × 2 (tree and non-tree)
  5. Build `pipelines` dict (6 models, each `Pipeline([prep, model])`)
  6. `train_candidates` (5-fold CV)
  7. `tune_model` (random search on best)
  8. `plot_keras_history` if Keras won
  9. `evaluate_model` → `save_best_model`
  10. `print_summary`

---

## What Was Removed
| Before | After |
|---|---|
| Single 60/20/20 holdout | 5-fold CV on train pool + 20% final test holdout |
| Encoding/scaling in `preprocess_data` | Encoding/scaling inside sklearn Pipelines |
| Only RMSE metric | RMSE, MSE, MAE, R² |
| No visualizations | Raw + clean data viz; fold score charts; Keras loss curve |
| No model save | Best model saved to `models/` |
| 5 sklearn models | 6 models (+ Keras NN) |
