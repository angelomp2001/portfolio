# libraries
import numpy as np
import time
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def model_training(
    features_train_regressions_scaled, target_train_reg_vectorized,
    df_valid_regressions,
    feature_train_ML_scaled, target_train_ML_vectorized,
    feature_valid_ML_scaled, target_valid_ML_vectorized,
    feature_test_ML_scaled, target_test_ML_vectorized,
):
    """
    Train and evaluate five regression models:
      1. LinearRegression  (regression-encoded, scaled features)
      2. LGBMRegressor     (ML-encoded, scaled features)
      3. RandomForestRegressor
      4. CatBoostRegressor  ← best model; also runs hyperparameter search
      5. XGBRegressor

    Prints RMSE, train time, and prediction time for each model.
    Final evaluation of CatBoost is run on the held-out test set.
    """

    # ── 1. Linear Regression ──────────────────────────────────────────────────
    lr_model = LinearRegression()

    start_time = time.time()
    lr_model.fit(features_train_regressions_scaled, target_train_reg_vectorized)
    lr_train_time = time.time() - start_time
    print(f'[LinearRegression] train time: {lr_train_time:.4f}s')

    start_time = time.time()
    target_valid_pred = lr_model.predict(features_train_regressions_scaled)
    lr_pred_time = time.time() - start_time
    print(f'[LinearRegression] pred time:  {lr_pred_time:.4f}s')

    linear_rmse = np.sqrt(mean_squared_error(target_train_reg_vectorized, target_valid_pred))
    print(f'[LinearRegression] RMSE: {linear_rmse:.4f}\n')

    # ── 2. LightGBM ───────────────────────────────────────────────────────────
    lgb_model = lgb.LGBMRegressor()

    start_time = time.time()
    lgb_model.fit(feature_train_ML_scaled, target_train_ML_vectorized)
    lgb_train_time = time.time() - start_time
    print(f'[LGBMRegressor] train time: {lgb_train_time:.4f}s')

    start_time = time.time()
    target_valid_ML_pred = lgb_model.predict(feature_valid_ML_scaled)
    lgb_pred_time = time.time() - start_time
    print(f'[LGBMRegressor] pred time:  {lgb_pred_time:.4f}s')

    lgb_rmse = np.sqrt(mean_squared_error(target_valid_ML_vectorized, target_valid_ML_pred))
    print(f'[LGBMRegressor] RMSE: {lgb_rmse:.4f}\n')

    # ── 3. Random Forest ──────────────────────────────────────────────────────
    rfr_model = RandomForestRegressor()

    start_time = time.time()
    rfr_model.fit(feature_train_ML_scaled, target_train_ML_vectorized)
    rfr_train_time = time.time() - start_time
    print(f'[RandomForestRegressor] train time: {rfr_train_time:.4f}s')

    start_time = time.time()
    target_valid_ML_pred = rfr_model.predict(feature_valid_ML_scaled)
    rfr_pred_time = time.time() - start_time
    print(f'[RandomForestRegressor] pred time:  {rfr_pred_time:.4f}s')

    rfr_rmse = np.sqrt(mean_squared_error(target_valid_ML_vectorized, target_valid_ML_pred))
    print(f'[RandomForestRegressor] RMSE: {rfr_rmse:.4f}\n')

    # ── 4. CatBoost ───────────────────────────────────────────────────────────
    cat_model = cb.CatBoostRegressor(verbose=0)

    start_time = time.time()
    cat_model.fit(feature_train_ML_scaled, target_train_ML_vectorized)
    cat_train_time = time.time() - start_time
    print(f'[CatBoostRegressor] train time: {cat_train_time:.4f}s')

    start_time = time.time()
    target_valid_ML_pred = cat_model.predict(feature_valid_ML_scaled)
    cat_pred_time = time.time() - start_time
    print(f'[CatBoostRegressor] pred time:  {cat_pred_time:.4f}s')

    cat_rmse = np.sqrt(mean_squared_error(target_valid_ML_vectorized, target_valid_ML_pred))
    print(f'[CatBoostRegressor] RMSE: {cat_rmse:.4f}\n')

    # ── 4a. CatBoost — max_depth search ───────────────────────────────────────
    print('[CatBoost] Searching max_depth in range(2, 6)...')
    for value in range(2, 6):
        model = cb.CatBoostRegressor(max_depth=value, verbose=0)
        model.fit(feature_train_ML_scaled, target_train_ML_vectorized)
        pred = model.predict(feature_valid_ML_scaled)
        rmse = np.sqrt(mean_squared_error(target_valid_ML_vectorized, pred))
        print(f'  max_depth={value}  RMSE={rmse:.4f}')  # best: 4

    # ── 4b. CatBoost — n_estimators search (max_depth fixed at 4) ─────────────
    print('\n[CatBoost] Searching n_estimators in range(50, 100) with max_depth=4...')
    for value in range(50, 100):
        model = cb.CatBoostRegressor(n_estimators=value, max_depth=4, verbose=0)
        model.fit(feature_train_ML_scaled, target_train_ML_vectorized)
        pred = model.predict(feature_valid_ML_scaled)
        rmse = np.sqrt(mean_squared_error(target_valid_ML_vectorized, pred))
        print(f'  n_estimators={value}  RMSE={rmse:.4f}')  # best: 87

    # ── 5. XGBoost ────────────────────────────────────────────────────────────
    xgb_model = xgb.XGBRegressor()

    start_time = time.time()
    xgb_model.fit(feature_train_ML_scaled, target_train_ML_vectorized)
    xgb_train_time = time.time() - start_time
    print(f'\n[XGBRegressor] train time: {xgb_train_time:.4f}s')

    start_time = time.time()
    target_valid_ML_pred = xgb_model.predict(feature_valid_ML_scaled)
    xgb_pred_time = time.time() - start_time
    print(f'[XGBRegressor] pred time:  {xgb_pred_time:.4f}s')

    xgb_rmse = np.sqrt(mean_squared_error(target_valid_ML_vectorized, target_valid_ML_pred))
    print(f'[XGBRegressor] RMSE: {xgb_rmse:.4f}\n')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('=' * 60)
    print('MODEL COMPARISON SUMMARY')
    print('=' * 60)
    print(f'  LinearRegression    RMSE={linear_rmse:.2f}  train={lr_train_time:.3f}s  pred={lr_pred_time:.4f}s')
    print(f'  LGBMRegressor       RMSE={lgb_rmse:.2f}  train={lgb_train_time:.3f}s  pred={lgb_pred_time:.4f}s')
    print(f'  RandomForest        RMSE={rfr_rmse:.2f}  train={rfr_train_time:.3f}s  pred={rfr_pred_time:.4f}s')
    print(f'  CatBoostRegressor   RMSE={cat_rmse:.2f}  train={cat_train_time:.3f}s  pred={cat_pred_time:.4f}s  ← best')
    print(f'  XGBRegressor        RMSE={xgb_rmse:.2f}  train={xgb_train_time:.3f}s  pred={xgb_pred_time:.4f}s')

    # ── Final test evaluation (best model: CatBoost tuned) ────────────────────
    final_model = cb.CatBoostRegressor(n_estimators=87, max_depth=4, verbose=0)
    final_model.fit(feature_train_ML_scaled, target_train_ML_vectorized)

    start_time = time.time()
    target_test_ML_pred = final_model.predict(feature_test_ML_scaled)
    cat_pred_time = time.time() - start_time

    cat_test_rmse = np.sqrt(mean_squared_error(target_test_ML_vectorized, target_test_ML_pred))
    print(f'\n[Final CatBoost (n_estimators=87, max_depth=4)]')
    print(f'  Test RMSE: {cat_test_rmse:.4f}  pred time: {cat_pred_time:.4f}s')
