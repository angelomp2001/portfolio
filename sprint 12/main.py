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

from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_candidates, tune_model, evaluate_model, print_summary

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = 'data/car_data.csv'
SAMPLE_SIZE  = 10_000   # Full dataset is ~350k rows; a sample keeps development fast.
RANDOM_STATE = 12345    # Fixed seed ensures reproducible splits across runs.
N_ITER_TUNE  = 20       # Number of random parameter combos to try during tuning.

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data(DATA_PATH)
df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
print(f"Loaded data — shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# ── Preprocess ────────────────────────────────────────────────────────────────
# preprocess_data handles every step needed before a model can be trained:
#   1. Drop columns that are irrelevant or would cause data leakage.
#   2. Clean numeric columns by nullifying out-of-range values.
#   3. Fill NaN in categorical columns with a literal 'missing' category so
#      those rows aren't lost at the dropna step.
#   4. Deduplicate and drop any remaining NaN rows.
#   5. Split into train / validation / test sets (60 / 20 / 20).
#   6. Encode categoricals two ways:
#        - One-Hot Encoding  → for linear regression models (avoids ordinality).
#        - Label Encoding    → for tree-based ML models (handles high cardinality better).
#   7. Scale all features with StandardScaler so distance-sensitive models
#      treat each feature equally (fit on train only to prevent data leakage).
#
# Returns a nested dict so callers can pick exactly what they need by name.
data = preprocess_data(df)

# Unpack regression (OHE) and ML (label-encoded) splits for clarity below.
reg = data['regression']
ml  = data['ml']

# ── Define candidate models ───────────────────────────────────────────────────
# Each entry specifies the model and the exact data split it should train on.
# LinearRegression uses OHE-encoded features; all tree models use label-encoded features.
candidates = {
    "LinearRegression": {
        "model":   LinearRegression(),
        "X_train": reg['scaled']['features']['train'],
        "y_train": reg['scaled']['targets']['train'],
        "X_valid": reg['scaled']['features']['valid'],
        "y_valid": reg['scaled']['targets']['valid'],
        "X_test":  reg['scaled']['features']['test'],
        "y_test":  reg['scaled']['targets']['test'],
    },
    "LGBMRegressor": {
        "model":   lgb.LGBMRegressor(verbose=-1),
        "X_train": ml['scaled']['features']['train'],
        "y_train": ml['scaled']['targets']['train'],
        "X_valid": ml['scaled']['features']['valid'],
        "y_valid": ml['scaled']['targets']['valid'],
        "X_test":  ml['scaled']['features']['test'],
        "y_test":  ml['scaled']['targets']['test'],
    },
    "RandomForestRegressor": {
        "model":   RandomForestRegressor(),
        "X_train": ml['scaled']['features']['train'],
        "y_train": ml['scaled']['targets']['train'],
        "X_valid": ml['scaled']['features']['valid'],
        "y_valid": ml['scaled']['targets']['valid'],
        "X_test":  ml['scaled']['features']['test'],
        "y_test":  ml['scaled']['targets']['test'],
    },
    "CatBoostRegressor": {
        "model":   cb.CatBoostRegressor(verbose=0),
        "X_train": ml['scaled']['features']['train'],
        "y_train": ml['scaled']['targets']['train'],
        "X_valid": ml['scaled']['features']['valid'],
        "y_valid": ml['scaled']['targets']['valid'],
        "X_test":  ml['scaled']['features']['test'],
        "y_test":  ml['scaled']['targets']['test'],
    },
    "XGBRegressor": {
        "model":   xgb.XGBRegressor(),
        "X_train": ml['scaled']['features']['train'],
        "y_train": ml['scaled']['targets']['train'],
        "X_valid": ml['scaled']['features']['valid'],
        "y_valid": ml['scaled']['targets']['valid'],
        "X_test":  ml['scaled']['features']['test'],
        "y_test":  ml['scaled']['targets']['test'],
    },
}

# ── Train & compare all candidates ───────────────────────────────────────────
results, best_name = train_candidates(candidates)

# ── Tune the best model (random search; skipped if no param grid exists) ─────
best_model, best_params = tune_model(
    results[best_name]["model"],
    candidates[best_name]["X_train"], candidates[best_name]["y_train"],
    candidates[best_name]["X_valid"], candidates[best_name]["y_valid"],
    n_iter=N_ITER_TUNE, random_state=RANDOM_STATE,
)

# ── Evaluate tuned model on held-out test data ───────────────────────────────
test_rmse, test_pred_time = evaluate_model(
    best_model,
    candidates[best_name]["X_test"],
    candidates[best_name]["y_test"],
)

# ── Summary ───────────────────────────────────────────────────────────────────
print_summary(results, best_name, best_params, test_rmse, test_pred_time)

# ── Keep all chart windows open ───────────────────────────────────────────────
# plt.show(block=True) blocks here until the user manually closes every window.
input("\nPress Enter to close charts and exit…")
plt.close("all")
