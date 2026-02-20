"""
Business case: Client wants to offer their customers an automated estimate of what their car could sell for.
Project goal: Predict 'Price' using structured car listing data.
Best model:   CatBoostRegressor (lowest RMSE after hyperparameter tuning)
"""

from src.data_preprocessing import load_data, preprocess_data
from src.model_training import model_training

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = 'data/car_data.csv'
SAMPLE_SIZE = 10_000
RANDOM_STATE = 12345

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data(DATA_PATH)
df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
print(f"Loaded data — shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# ── Preprocess ────────────────────────────────────────────────────────────────
(
    df, ordinal, categorical,
    df_train_regressions, df_valid_regressions, df_test_regressions,
    df_train_ML, df_valid_ML, df_test_ML,
    df_train_regressions_scaled, df_valid_regressions_scaled, df_test_regressions_scaled,
    df_train_ML_scaled, df_valid_ML_scaled, df_test_ML_scaled,
    features_train_regressions_scaled, feature_valid_regressions_scaled, feature_test_regressions_scaled,
    feature_train_ML_scaled, feature_valid_ML_scaled, feature_test_ML_scaled,
    target_train_reg_vectorized, target_valid_reg_vectorized, target_test_reg_vectorized,
    target_train_ML_vectorized, target_valid_ML_vectorized, target_test_ML_vectorized,
) = preprocess_data(df)

# ── Train & Evaluate ──────────────────────────────────────────────────────────
model_training(
    features_train_regressions_scaled, target_train_reg_vectorized,
    df_valid_regressions,
    feature_train_ML_scaled, target_train_ML_vectorized,
    feature_valid_ML_scaled, target_valid_ML_vectorized,
    feature_test_ML_scaled, target_test_ML_vectorized,
)
