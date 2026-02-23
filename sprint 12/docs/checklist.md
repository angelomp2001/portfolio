Project Checklist:
[✅] Global random seed
[✅] Parameter grid
[✅] Hyperparameter grid
[✅] Save raw data statistics
[✅] Sampling (optional)
[✅] Visualize raw data: histogram/bar chart
[✅] Label numeric categorical and time data
[✅] Label target
[✅] Evaluation Metrics: MAE, MSE, RMSE, R2 (regression-relevant; classification metrics N/A)
[❌] Time data as datetime (date columns dropped as irrelevant to price)
[❌] Feature engineer datetime: hour/day/month/weekday/lag features (see above)
[✅] Apply PolynomialFeatures (degree=2 on numeric cols for non-tree models)
[✅] Clean data distribution
[✅] Clean data visualizations: univariate, bivariate, timeseries
[✅] Save clean data statistics
[✅] Training visualizations: timeseries (fold-score per fold; epoch vs loss for Keras)
[✅] training statistics: duration, Max memory usage (peak MB via tracemalloc)
[✅] RandomizedSearch
[✅] Cross validation with holdout: KFold(k=5) + final holdout test set
[❌] StratifiedKFold (classification only), TimeSeriesSplit (no time series target)
[✅] Pipeline (sklearn Pipeline for each model with preprocessor + model)
[✅] Leakage-safe Encoding: Ordinal for tree, OHE otherwise
[✅] Feature scaling: Min/max for bounded values, standardization otherwise
[✅] Early Stopping parameters (Keras: EarlyStopping patience=15)
[✅] Dropout parameter (Keras: Dropout(0.3) after Dense layers)
[✅] Model checkpoints (Keras: ModelCheckpoint saves best epoch on final fit)
[✅] Learning rate adaptation (Keras: ReduceLROnPlateau factor=0.5, patience=7)
[✅] Save best model (joblib for sklearn pipelines, .keras for NN)
[  ] API: load best model, run inference, log and save run and results
[  ] Unit testable
