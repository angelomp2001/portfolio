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
df = preprocess_datatime_series(df, column_names='datetime')

df = preprocess_num_orders(df, column_names=['num_orders'])

## Analysis
# Plot raw series
df_target = df['num_orders']
plot_raw_series(df)

# Plot Median
plot_median(df, df_target)


# Plot ACF
acf = acf_plot(df_target, lags_to_check=500)


# Plot PACF
pacf = pacf_plot(df_target, lags_to_check=500)


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

start_value = len(df_train)  # = test.index[0]  First index of the test set 
end_value = len(df_train) + len(df_test) - 1 # = test.index[-1] # Last index of the test set

# Eval AR model with p
ar_rmse_value = eval_ar(start_value, end_value, df_train, df_test, df_train_target, df_test_target, lag_ar)

# Fit MA model with q
ma_rmse_value = eval_ma(start_value, end_value, df_train, df_test, df_train_target, df_test_target, lag_ma)


# Eval ARMA model with p and q
ar_rmse_value = eval_arma(start_value, end_value, df_train, df_test, df_train_target, df_test_target, lag_ar, lag_ma)

# Score stationary (d)
df_stationarityTest = check_stationarity(df_train_target)

# Eval ARIMA model with p, d, q
arima_rmse_value = eval_arima(start_value, end_value, df_train, df_test, df_train_target, df_test_target, lag_ar, lag_ma)

# Plot decompose (trend, seasonality, residuals)
decomposition = seasonal_decompose(df_train_target)
decomposition.plot()
plt.show()

# Plot periodicity (decomposition.seasonal[0:months_in_period].plot())
N = 3
hours_per_day = 24
months_in_period = N * hours_per_day

plot_periodicity(decomposition, months_in_period)

# Estimate window of seasonality (s). ie. 12 if annual seasonality/cycle measured in months.
# 24 for hours in a day

# Eval SARIMAX:
rmse = eval_sarimax(df_train, df_test, df_train_target, df_test_target)

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