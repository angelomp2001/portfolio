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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    load_data, clean_data, build_preprocessor,
    save_data_stats, visualize_data,
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
K_FOLDS      = 5           # k for KFold CV
N_ITER_TUNE  = 20

# ── Column roles ──────────────────────────────────────────────────────────────
# cat/num feature columns are inferred from dtype
TARGET_COL   = 'Price'
COLS_TO_DROP = ['DateCrawled', 'RegistrationMonth', 'DateCreated',
                'NumberOfPictures', 'PostalCode', 'LastSeen']

# ── Value-range guards ────────────────────────────────────────────────────────
PRICE_MIN            = 500
YEAR_MIN, YEAR_MAX   = 1900, 2025
POWER_MIN, POWER_MAX = 100, 400

# ── Split config ──────────────────────────────────────────────────────────────
TEST_RATIO = 0.2

# ── Keras hyperparameters ─────────────────────────────────────────────────────
KERAS_EPOCHS           = 150
KERAS_BATCH_SIZE       = 64
KERAS_VALIDATION_SPLIT = 0.1
KERAS_DROPOUT_RATE     = 0.3
KERAS_LEARNING_RATE    = 0.001

# ── Models — Regression ───────────────────────────────────────────────────────
# Each entry: (name, model_instance, is_tree)
# is_tree=True  → OrdinalEncoding, no scaling (LGBM, RF, CatBoost, XGB)
# is_tree=False → OHE + PolynomialFeatures(degree=2) + StandardScaler (Linear, Keras NN)
MODELS = [
    ("LinearRegression",      LinearRegression(),                               False),
    ("KerasRegressorWrapper", KerasRegressorWrapper(
                                  build_fn=build_keras_model,
                                  epochs=KERAS_EPOCHS,
                                  batch_size=KERAS_BATCH_SIZE,
                                  validation_split=KERAS_VALIDATION_SPLIT,
                                  dropout_rate=KERAS_DROPOUT_RATE,
                                  learning_rate=KERAS_LEARNING_RATE,
                              ),                                                False),
    ("LGBMRegressor",         lgb.LGBMRegressor(verbose=-1),                   True),
    ("RandomForestRegressor", RandomForestRegressor(),                          True),
    ("CatBoostRegressor",     cb.CatBoostRegressor(verbose=0),                 True),
    ("XGBRegressor",          xgb.XGBRegressor(),                              True),
]

# ── Models — Classification (swap in when TARGET_COL is categorical) ──────────
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# MODELS_CLASSIFICATION = [
#     ("LogisticRegression",       LogisticRegression(max_iter=1000),            False),
#     ("LGBMClassifier",           lgb.LGBMClassifier(verbose=-1),               True),
#     ("RandomForestClassifier",   RandomForestClassifier(),                     True),
#     ("CatBoostClassifier",       cb.CatBoostClassifier(verbose=0),             True),
#     ("XGBClassifier",            xgb.XGBClassifier(),                         True),
# ]

# ── Load ──────────────────────────────────────────────────────────────────────
df_raw = load_data(DATA_PATH)
df_raw = df_raw.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
print(f"Loaded data — shape: {df_raw.shape}\n")

# ── Raw data: stats + visualizations ─────────────────────────────────────────
# save_data_stats(df_raw, 'data/stats_raw.json',   label='raw')
# visualize_data(df_raw,  label='Raw',  out_path='data/viz_raw.png')

# ── Clean ─────────────────────────────────────────────────────────────────────
df = clean_data(
    df_raw,
    target_col=TARGET_COL,
    cols_to_drop=COLS_TO_DROP,
    price_min=PRICE_MIN,
    year_min=YEAR_MIN, year_max=YEAR_MAX,
    power_min=POWER_MIN, power_max=POWER_MAX,
)
print(f"After cleaning — shape: {df.shape}\n")

# ── Clean data: stats + visualizations ───────────────────────────────────────
# save_data_stats(df, 'data/stats_clean.json', label='clean')
# visualize_data(df,  label='Clean', out_path='data/viz_clean.png')

# ── Split: 80% train-pool / 20% final holdout test ───────────────────────────
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_RATIO, 
    random_state=RANDOM_STATE,
)
print(f"Train pool: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows\n")

# ── Infer cat/num cols from dtype (after cleaning, target already dropped) ────
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
num_cols = X_train.select_dtypes(include='number').columns.tolist()

# ── Build pipelines (for loop — every model goes through the same pattern) ────
# Non-tree  → OHE + PolynomialFeatures(degree=2) + StandardScaler  ✅
# Tree      → OrdinalEncoding, no scaling                           ✅
pipelines = {}
for name, model, is_tree in MODELS:
    pipelines[name] = Pipeline([
        ("prep",  build_preprocessor(cat_cols, num_cols, is_tree=is_tree)),
        ("model", model),
    ])

# ── K-fold CV: train & compare all candidates ─────────────────────────────────
results, best_name = train_candidates(
    pipelines, X_train, y_train,
    k_folds=K_FOLDS, random_state=RANDOM_STATE,
)

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
test_metrics = evaluate_model(best_pipeline, X_test, y_test)
save_best_model(best_pipeline, best_name, best_params, test_metrics)

# ── Print summary ─────────────────────────────────────────────────────────────
print_summary(results, best_name, best_params, test_metrics)

# ── Keep all chart windows open ───────────────────────────────────────────────
input("\nPress Enter to close charts and exit…")
plt.close("all")
