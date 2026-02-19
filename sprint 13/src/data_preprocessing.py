#libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import arma_order_select_ic
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX



def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['datetime'])

def preprocess_datatime_series(df, column_names='datetime'):
    # set index to dt
    df[column_names] = pd.to_datetime(df[column_names])
    df.set_index(df[column_names], inplace=True)
    df.drop(columns=column_names, inplace=True)

    return df

def preprocess_num_orders(df, column_names=['num_orders']):
    # group to 1 hour
    df = df.resample('1H').sum()
    df[column_names].plot()
    print(df.head())

    # df_train: Create lag features
    # df_train: Create calendar features
    df = (df
            .assign(day=df.index.day,
                    month=df.index.month,
                    dayofweek=df.index.dayofweek))

    #QC
    print(df.head())

    # Dropna
    df = df.dropna(axis=0, how='all')
    return df

def plot_raw_series(df):
    df.plot(figsize=(20, 6))
    plt.show()

def plot_median(df, df_target):
    df_target_median = df_target.median()
    pred_median = np.ones(len(df)) * df_target_median
    plt.plot(figsize=(20, 6))
    plt.plot(df.index, df_target, label='Raw Data', color='blue')
    plt.plot(df.index, pred_median, label='Median', color='red')

    # Score RMSE (24)
    print('RMSE:', np.sqrt(mean_squared_error(df_target, pred_median))) # RMSE: 45.474613179669596

def acf_plot(df_target, lags_to_check=500):
    lags_to_check = 500
    fig, ax = plt.subplots(figsize=(20, 6)) 
    acf = plot_acf(x=df_target, lags=lags_to_check, ax=ax)
    plt.xlabel("Lags")
    plt.ylabel("ACF")
    plt.show()

    return acf

def pacf_plot(df_target, lags_to_check=500):
    fig, ax = plt.subplots(figsize=(20, 6)) 
    pacf = plot_pacf(x=df_target, lags=lags_to_check, ax=ax)
    plt.xlabel("Lags")
    plt.ylabel("PACF")
    plt.show()

    return pacf

def eval_ar(start_value, end_value, df_train, df_test, df_train_target, df_test_target, lag_ar):
    # Fit AR model with p
    ar_model = AutoReg(df_train_target, lags=lag_ar, seasonal=True)
    ar_model = ar_model.fit()

    # Predict AR
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

    return ar_rmse_value

def eval_ma(start_value, end_value, df_train, df_test, df_train_target, df_test_target, lag_ma):
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
    ma_rmse_value = np.sqrt(mean_squared_error(df_test_target, ma_pred))
    print(ma_rmse_value.round(3)) # 70.459

    return ma_rmse_value

def eval_arma(start_value, end_value, df_train, df_test, df_train_target, df_test_target, lag_ma, lag_ar):
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

    return ar_rmse_value

def check_stationarity(df_train_target):
    # run the adfuller test to check for stationarity
    df_stationarityTest = adfuller(df_train_target, autolag='AIC')
    print("P-value: ", df_stationarityTest[1])

    return df_stationarityTest[1]

def eval_arima(start_value, end_value, df_train, df_test, df_train_target, df_test_target, lag_ma, lag_ar):
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
    arima_rmse_value = np.sqrt(mean_squared_error(df_test_target, arima_pred))
    print(arima_rmse_value.round(3)) # RMSE: 52.218

    return arima_rmse_value

def plot_periodicity(decomposition, months_in_period):
    decomposition.seasonal[0:months_in_period].plot()
    plt.show()

def eval_sarimax(df_train, df_test, df_train_target, df_test_target):
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

    return rmse