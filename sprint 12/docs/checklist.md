Project Checklist:
✅ Global random seed
✅ Parameter grid
✅ Hyperparameter grid
✅ Save raw data statistics
✅ Sampling (optional)
[ ] Visualize raw data: histogram/bar chart
[ ] Label numeric categorical and time data
[ ] Label target
[ ] Evaluation Metrics: MAE, MSE, RMSE, R2, Accuracy, Precision, recall, F1, Confusion matrix, ROC AUC, PR AUC, Log loss
[ ] Time data as datetime
[ ] Feature engineer datetime: hour/day/month/weekday/lag features
[ ] Apply PolynomialFeatures
[ ] Clean data distribution
[ ] Clean data visualizations: univariate, bivariate, timeseries
✅ Save clean data statistics
[ ] Training visualizations: timeseries
[ ] training statistics: duration, Max memory usage, CPU/GPU utilization
✅ RandomizedSearch
[ ] Cross validation with holdout: KFold, StratifiedKFold, TimeSeriesSplit
[ ] Pipeline
✅ Leakage-safe Encoding: Ordinal for tree, OHE otherwise
✅ Feature scaling: Min/max for bounded values, standardization otherwise. 
[ ] Early Stopping parameters
[ ] Dropout parameter
[ ] Model checkpoints
[ ] Learning rate adaptation
[ ] Save best model
[ ] API: load best model, run inference, log and save run and results
[ ] Unit testable
