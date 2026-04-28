import numpy as np
import pandas as pd
import time
import os
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, mean_absolute_error

from prepare import prepare_ticker, TICKERS, LOOK_BACK

# -------------------------------------------------------
# EXPERIMENT: GRU 2-layer look_back=30 + lag_2, units lebih besar (256, 128)
# Hipotesis: GRU lb=30 + lag_2 adalah best so far (avg=114.01).
# BBCA masih tinggi (171). Coba GRU dengan unit lebih besar (256, 128)
# untuk kapasitas lebih tinggi dalam menangkap pola BBCA.
# -------------------------------------------------------

EXPERIMENT_LOOK_BACK = 30
EXPERIMENT_SLUG = "gru2l_lb30_256_128"
FEATURES = ["close", "RSI_14", "MA_20", "log_return", "lag_1", "lag_2"]

def get_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except:
        return "unknown_commit"

def build_model(input_shape):
    """
    2-layer GRU dengan units lebih besar:
    GRU(256, return_sequences=True) -> Dropout(0.2) -> GRU(128) -> Dropout(0.2) -> Dense(1)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(256, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(128),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )
    return model

def prepare_ticker_extended(ticker, look_back, features):
    import polars as pl
    import yfinance as yf
    from sklearn.preprocessing import MinMaxScaler

    df = yf.download(ticker, period="2y", interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df_pl = pl.from_pandas(df)

    df_pl = df_pl.with_columns([
        pl.col("close").shift(1).alias("lag_1"),
        pl.col("close").shift(2).alias("lag_2"),
        pl.col("close").rolling_mean(20).alias("MA_20"),
    ])
    df_pl = df_pl.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")
    )
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0).rolling_mean(14)
    loss = (-delta).clip(lower_bound=0).rolling_mean(14)
    rs = gain / loss
    df_pl = df_pl.with_columns((1 - (1 / (1 + rs))).alias("RSI_14"))
    df_pl = df_pl.drop_nulls()

    df_pd = df_pl.select(features).to_pandas()
    data = df_pd.values

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

    def make_sequences(d, lb):
        X, y = [], []
        for i in range(lb, len(d)):
            X.append(d[i-lb:i])
            y.append(d[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = make_sequences(train, look_back)
    X_val, y_val     = make_sequences(val, look_back)
    X_test, y_test   = make_sequences(test, look_back)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def train_one_ticker(ticker, commit_hash):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_ticker_extended(
        ticker, look_back=EXPERIMENT_LOOK_BACK, features=FEATURES
    )

    model = build_model((X_train.shape[1], X_train.shape[2]))

    es = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    y_pred = model.predict(X_test).flatten()

    n_features = scaler.n_features_in_
    dummy = np.zeros((len(y_test), n_features))
    dummy[:, 0] = y_test
    y_test_inv = scaler.inverse_transform(dummy)[:, 0]

    dummy[:, 0] = y_pred
    y_pred_inv = scaler.inverse_transform(dummy)[:, 0]

    rmse = mean_squared_error(y_test_inv, y_pred_inv) ** 0.5
    mae  = mean_absolute_error(y_test_inv, y_pred_inv)

    out_dir = f"evaluation_output/{commit_hash}"
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{ticker} Training History -- GRU(256,128) lb=30")
    plt.legend()
    plt.savefig(f"{out_dir}/training_history_{ticker}.png")
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(y_test_inv, label="Actual (Groundtruth)", color="steelblue")
    plt.plot(y_pred_inv, label="Predicted", color="tomato", linestyle="--")
    plt.title(f"GRU(256,128) lb=30 -- {ticker} | RMSE={rmse:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/prediction_{ticker}.png")
    plt.close()

    model.save(f"{out_dir}/model_{ticker}_{EXPERIMENT_SLUG}.keras")

    return rmse, mae, len(history.history['loss'])

if __name__ == "__main__":
    t0 = time.time()
    commit_hash = get_commit_hash()
    out_dir = f"evaluation_output/{commit_hash}"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    rmses = []
    epochs_list = []

    for ticker in TICKERS:
        rmse, mae, epochs_run = train_one_ticker(ticker, commit_hash)
        results[ticker] = (rmse, mae)
        rmses.append(rmse)
        epochs_list.append(epochs_run)

        print("---")
        print(f"ticker:          {ticker}")
        print(f"rmse:            {rmse:.6f}")
        print(f"mae:             {mae:.6f}")

    rmse_avg = np.mean(rmses)
    elapsed = time.time() - t0
    print("---")
    print(f"rmse_avg:        {rmse_avg:.6f}")
    print(f"training_seconds:{elapsed:.1f}")
    print(f"model:           GRU(256,128) look_back={EXPERIMENT_LOOK_BACK} slug={EXPERIMENT_SLUG}")

    report_data = []
    for ticker, (rmse, mae) in results.items():
        report_data.append({"Ticker": ticker, "RMSE": rmse, "MAE": mae})
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(f"{out_dir}/regression_report.csv", index=False)
    try:
        df_report.to_excel(f"{out_dir}/regression_report.xlsx", index=False)
    except Exception:
        print("Note: openpyxl not available, skipping .xlsx")

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    card_content = f"""=== EXPERIMENT CARD ===
commit:         {commit_hash}
date:           {date_str}

--- Model ---
type:           GRU (2 layer, larger units 256/128)
architecture:   GRU(256, ret_seq=True) -> Dropout(0.2) -> GRU(128) -> Dropout(0.2) -> Dense(1)
optimizer:      Adam lr=0.001
epochs_run:     {np.mean(epochs_list):.1f} avg (max=100, EarlyStopping patience=10)

--- Preprocessing ---
scaler:         MinMaxScaler (fit on train only)
look_back:      {EXPERIMENT_LOOK_BACK}
use_log_return: False
features:       close, RSI_14, MA_20, log_return, lag_1, lag_2

--- Results ---
BBCA: rmse={results['BBCA.JK'][0]:.2f}, mae={results['BBCA.JK'][1]:.2f}
BBRI: rmse={results['BBRI.JK'][0]:.2f}, mae={results['BBRI.JK'][1]:.2f}
BMRI: rmse={results['BMRI.JK'][0]:.2f}, mae={results['BMRI.JK'][1]:.2f}
rmse_avg: {rmse_avg:.2f}
status:   (to be determined)

--- Why tried ---
GRU lb=30 lag_2 best so far (114.01). Coba GRU units lebih besar (256, 128) untuk BBCA.

--- What worked / didn't ---
(to be filled after run)
"""
    with open(f"{out_dir}/experiment_card.txt", "w", encoding="utf-8") as f:
        f.write(card_content)