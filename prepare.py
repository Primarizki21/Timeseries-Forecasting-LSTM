# prepare.py (NEW — STOCK FORECASTING)

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

TICKERS = ["BBCA.JK", "BBRI.JK", "BMRI.JK"]
LOOK_BACK = 20  # default baseline

# --------------------------------------------------
# Fetch data (2 tahun terakhir)
# --------------------------------------------------
def fetch_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return pl.from_pandas(df)

# --------------------------------------------------
# Feature Engineering (Polars)
# --------------------------------------------------
def add_features(df):
    df = df.with_columns([
        pl.col("close").shift(1).alias("lag_1"),
        pl.col("close").shift(2).alias("lag_2"),
        pl.col("close").rolling_mean(20).alias("MA_20"),
    ])

    # log return
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")
    )

    # RSI
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0).rolling_mean(14)
    loss = (-delta).clip(lower_bound=0).rolling_mean(14)
    rs = gain / loss

    df = df.with_columns(
        (1 - (1 / (1 + rs))).alias("RSI_14")
    )

    return df.drop_nulls()

# --------------------------------------------------
# Split + scaling + sequence
# --------------------------------------------------
def make_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, 0])  # close
    return np.array(X), np.array(y)

def prepare_ticker(ticker, look_back=LOOK_BACK):
    df = fetch_data(ticker)
    df = add_features(df)

    features = ["close", "RSI_14", "MA_20", "log_return", "lag_1"]
    df_pd = df.select(features).to_pandas()

    data = df_pd.values

    # split (chronological)
    n = len(data)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)

    train = data[:train_end]
    val   = data[train_end:val_end]
    test  = data[val_end:]

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    val   = scaler.transform(val)
    test  = scaler.transform(test)

    X_train, y_train = make_sequences(train, look_back)
    X_val, y_val     = make_sequences(val, look_back)
    X_test, y_test   = make_sequences(test, look_back)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler