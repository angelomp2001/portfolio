"""
Business case: Client wants to offer their customers an automated estimate
               of what their car could sell for.
Project goal:  Predict 'Price' using structured car listing data.
"""

import matplotlib.pyplot as plt
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    load_data, clean_data, split_data, build_preprocessor,
    save_data_stats, visualize_data,
    CATEGORICAL_COLS, NUMERIC_COLS,
)
from src.model_training import (
    KerasRegressorWrapper, build_keras_model,
    train_candidates, tune_model, evaluate_model,
    save_best_model, print_summary, plot_keras_history,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = 'data/car_data.csv'
SAMPLE_SIZE  = 10_000
RANDOM_STATE = 12345
N_ITER_TUNE  = 20

# ── Load ──────────────────────────────────────────────────────────────────────
df_raw = load_data(DATA_PATH)
df_raw = df_raw.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
print(f"Loaded data — shape: {df_raw.shape}\n")

# ── Raw data: stats + visualizations ─────────────────────────────────────────
save_data_stats(df_raw, 'data/stats_raw.json',   label='raw')
visualize_data(df_raw,  label='Raw',  out_path='data/viz_raw.png')

# ── Clean ─────────────────────────────────────────────────────────────────────
df = clean_data(df_raw)
print(f"After cleaning — shape: {df.shape}\n")

# ── Clean data: stats + visualizations ───────────────────────────────────────
save_data_stats(df, 'data/stats_clean.json', label='clean')
visualize_data(df,  label='Clean', out_path='data/viz_clean.png')

# ── Split: 80% train-pool / 20% final holdout test ───────────────────────────
X_train, X_test, y_train, y_test = split_data(df)
print(f"Train pool: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows\n")

# ── Build preprocessors ───────────────────────────────────────────────────────
# Column lists inferred from the cleaned DataFrame (order matches build_preprocessor)
cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
num_cols = [c for c in NUMERIC_COLS     if c in X_train.columns]

prep_reg  = build_preprocessor(cat_cols, num_cols, is_tree=False)  # OHE + PolyFeat + Scale
prep_tree = build_preprocessor(cat_cols, num_cols, is_tree=True)   # OrdinalEnc + passthrough

# ── Build pipelines ───────────────────────────────────────────────────────────
# Each pipeline = preprocessor + model.
# Non-tree models get OHE + PolynomialFeatures(degree=2) on numeric cols. ✅
# Tree models get OrdinalEncoding (no scaling needed).                     ✅
# Keras NN uses the same non-tree preprocessor as LinearRegression.        ✅
pipelines = {
    "LinearRegression": Pipeline([
        ("prep",  prep_reg),
        ("model", LinearRegression()),
    ]),
    "KerasRegressorWrapper": Pipeline([
        ("prep",  build_preprocessor(cat_cols, num_cols, is_tree=False)),
        ("model", KerasRegressorWrapper(
            build_fn=build_keras_model,
            epochs=150, batch_size=64,
            validation_split=0.1,
            dropout_rate=0.3,
            learning_rate=0.001,
        )),
    ]),
    "LGBMRegressor": Pipeline([
        ("prep",  build_preprocessor(cat_cols, num_cols, is_tree=True)),
        ("model", lgb.LGBMRegressor(verbose=-1)),
    ]),
    "RandomForestRegressor": Pipeline([
        ("prep",  build_preprocessor(cat_cols, num_cols, is_tree=True)),
        ("model", RandomForestRegressor()),
    ]),
    "CatBoostRegressor": Pipeline([
        ("prep",  build_preprocessor(cat_cols, num_cols, is_tree=True)),
        ("model", cb.CatBoostRegressor(verbose=0)),
    ]),
    "XGBRegressor": Pipeline([
        ("prep",  build_preprocessor(cat_cols, num_cols, is_tree=True)),
        ("model", xgb.XGBRegressor()),
    ]),
}

# ── 5-fold CV: train & compare all candidates ─────────────────────────────────
results, best_name = train_candidates(pipelines, X_train, y_train)

# ── Tune the best model (random search with live chart) ───────────────────────
best_pipeline, best_params = tune_model(
    pipelines[best_name], best_name,
    X_train, y_train,
    n_iter=N_ITER_TUNE, random_state=RANDOM_STATE,
)

# ── Plot Keras training curve if winner is the NN ─────────────────────────────
final_step = best_pipeline.steps[-1][1]
if isinstance(final_step, KerasRegressorWrapper) and final_step.history_ is not None:
    plot_keras_history(final_step.history_, name=best_name)

# ── Save best model ───────────────────────────────────────────────────────────
# Evaluate first so we have metrics for the metadata file
test_metrics = evaluate_model(best_pipeline, X_test, y_test)
save_best_model(best_pipeline, best_name, best_params, test_metrics)

# ── Print summary ─────────────────────────────────────────────────────────────
print_summary(results, best_name, best_params, test_metrics)

# ── Keep all chart windows open ───────────────────────────────────────────────
input("\nPress Enter to close charts and exit…")
plt.close("all")
