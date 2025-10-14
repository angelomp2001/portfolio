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
    return pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

def preprocess_data(df):
    # set index to dt
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index(df['datetime'], inplace=True)
    df.drop(columns='datetime', inplace=True)

    # group to 1 hour
    print(df.head())
    df['num_orders'].plot()
    plt.show()

    df = df.resample('1H').sum()
    df['num_orders'].plot()
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

