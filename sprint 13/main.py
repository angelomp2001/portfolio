'''
Time series: Predict tomorrow's demand for taxis

Sweet Lift Taxi company has collected historical data on taxi orders at airports. To attract more drivers during peak hours, we need to predict the amount of taxi orders for the next hour. 
'''

from src.data_preprocessing import *

# load and view data
path = 'data/taxi.csv'
df = load_data(path)

# QC
print(df.shape)
print(df.head())
print(df.tail())
print(df.describe())
print(df.isna().sum())
print(df.columns)
df['num_orders'].plot()

# processing data
df = preprocess_data(df)

## Analysis
# Plot raw series
df_target = df['num_orders']
df.plot(figsize=(20, 6))
plt.show()

# Plot Median
df_target_median = df_target.median()
pred_median = np.ones(len(df)) * df_target_median
plt.plot(figsize=(20, 6))
plt.plot(df.index, df_target, label='Raw Data', color='blue')
plt.plot(df.index, pred_median, label='Median', color='red')

# Score RMSE (24)
print('RMSE:', np.sqrt(mean_squared_error(df_target, pred_median))) # RMSE: 45.474613179669596

# Plot ACF
lags_to_check = 500
fig, ax = plt.subplots(figsize=(20, 6)) 
acf = plot_acf(x=df_target, lags=lags_to_check, ax=ax)
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.show()


# Plot PACF
fig, ax = plt.subplots(figsize=(20, 6)) 
pacf = plot_pacf(x=df_target, lags=lags_to_check, ax=ax)
plt.xlabel("Lags")
plt.ylabel("PACF")
plt.show()


# QC: arma_order_select_ic() function
# df: find optimal number of lags
res = arma_order_select_ic(y=df_target, max_ar=24, max_ma=0)
print("arma_order_select_ic():", res.bic_min_order)
# arma_order_select_ic(): (24, 0)

lag_ar = 24

# df: find optimal number of ma
res = arma_order_select_ic(y=df_target, max_ar=0, max_ma=24)
print("arma_order_select_ic():", res.bic_min_order)
# arma_order_select_ic(): (0, 24)


lag_ma = 24

# Analsis results
'''
calendar lags: don't seem to coincide with spikes
Median: looks extremely far from test values
ACF chart: spikes tend exceed shared region for the whole 500 lags
PACF chart: spikes tend exceed shared region for the whole 500 lags
'''

# Training
# Split data (90 past /10 future)
df_train, df_test = train_test_split(df, shuffle=False, test_size=0.1, random_state = 12345)

# define target
df_train_target = df_train['num_orders']
df_test_target = df_test['num_orders']

# Fit AR model with p
ar_model = AutoReg(df_train_target, lags=lag_ar, seasonal=True)
ar_model = ar_model.fit()

# Predict AR
start_value = len(df_train)  # = test.index[0]  First index of the test set 
end_value = len(df_train) + len(df_test) - 1 # = test.index[-1] # Last index of the test set
ar_pred = ar_model.predict(start=start_value, end=end_value, dynamic=False)

# Plot AR
plt.figure(figsize=(20, 6))
plt.plot(df_test_target.index, ar_pred, color='blue', label='target_pred')
plt.plot(df_test_target.index, df_test_target, color='red', label='target')
plt.legend(loc="upper left")
plt.xticks(rotation=90)
plt.show()

# Eval AR
ar_rmse_value = np.sqrt(mean_squared_error(df_test_target, ar_pred))
print(ar_rmse_value.round(3)) # 68.927

# fit MA model with q
ma_model = ARIMA(df_train_target, order=(lag_ma, 0, 0))
ma_model = ma_model.fit()

# predict MA
ma_train = ma_model.predict(start=0, end=len(df_train), dynamic=False)
ma_pred = ma_model.predict(start=start_value, end=end_value, dynamic=False)

# Plot MA (q)
plt.plot(df_test_target.index, ma_pred, color="blue", label="pred")
plt.plot(df_test_target.index, df_test_target, color="red", label="test")
plt.legend(loc="upper left")
plt.xticks(rotation=90)
plt.show()

# Eval MA
ar_rmse_value = np.sqrt(mean_squared_error(df_test_target, ma_pred))
print(ar_rmse_value.round(3)) # 70.459

# Fit ARMA model with (p,q)
arma_model = ARIMA(df_train_target, order=(lag_ma, 0, lag_ar))
arma_model = arma_model.fit()

# Predict ARMA
start_value = len(df_train)
end_value = len(df_train) + len(df_test) - 1
arma_pred = arma_model.predict(start=start_value, end=end_value, dynamic=False)

# Plot ARMA
plt.plot(df_test_target.index, arma_pred, color="blue", label="pred")
plt.plot(df_test_target.index, df_test_target, color="red", label="test")
plt.legend(loc="upper left")
plt.xticks(rotation=90)
plt.show()

# Eval MA
ar_rmse_value = np.sqrt(mean_squared_error(df_test_target, arma_pred))
print(ar_rmse_value.round(3)) # RMSE: 46.076


# Score stationary (d)
# run the adfuller test to check for stationarity
df_stationarityTest = adfuller(df_train_target, autolag='AIC')
print("P-value: ", df_stationarityTest[1])

# Fit ARIMA model (p,d,q)
arima_model = ARIMA(df_train_target, order=(lag_ma, 1 , lag_ar))
arima_model = arima_model.fit()

# Predict ARIMA
arima_pred = arima_model.predict(start=start_value, end=end_value, dynamic=False)

# Plot ARIMA
plt.plot(df_test_target.index, arima_pred, color='blue', label='pred')
plt.plot(df_test_target.index, df_test_target, color='red', label='test')
plt.legend(loc="upper left")
plt.xticks(rotation=90)
plt.show()

# Eval ARIMA
ar_rmse_value = np.sqrt(mean_squared_error(df_test_target, arima_pred))
print(ar_rmse_value.round(3)) # RMSE: 52.218

# Plot decompose (trend, seasonality, residuals)
decomposition = seasonal_decompose(df_train_target)
decomposition.plot()
plt.show()

# Plot periodicity (decomposition.seasonal[0:months_in_period].plot())
N = 3
hours_per_day = 24
months_in_period = N * hours_per_day

decomposition.seasonal[0:months_in_period].plot()
plt.show()

# Estimate window of seasonality (s). ie. 12 if annual seasonality/cycle measured in months.
# 24 for hours in a day

# Fit SARIMAX:
sarimax_model = SARIMAX(df_train_target, 
                       order=(1, 1, 1),          # Simple non-seasonal
                       seasonal_order=(1, 1, 1, 24))  # Simple seasonal with period 24
sarimax_fit = sarimax_model.fit()

# # Predict SARIMAX
predictions = sarimax_fit.predict(start=len(df_train), end=len(df_train)+len(df_test)-1, dynamic=False)
df_test['predicted_num_orders'] = predictions

# plot the results
plt.figure(figsize=(12,6))
plt.plot(df_train.index, df_train_target, label="Train")
plt.plot(df_test.index, df_test_target, label="Test", color="blue")
plt.plot(df_test.index, predictions, label="Predictions", color="red")
plt.legend()
plt.show()

# # Eval SARIMAX
rmse = np.sqrt(mean_squared_error(df_test_target, predictions))
print(f"Test RMSE: {rmse:.2f}") # RMSE: 44.46

# get info about the model
sarimax_fit.summary()

# results

'''
RMSE: 
Median: 45.474613179669596 
AR: 68.927 
MA: 70.459 
ARMA: 46.076 
ARIMA: 52.218 
SARIMAX: 44.46

SARIMAX scored the best, but just calculating the median got you almost the same score, so just use the median.
'''