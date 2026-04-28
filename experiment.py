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

def get_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except:
        return "unknown_commit"

def build_model(input_shape):
    model = models.Sequential([
        layers.LSTM(64, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )
    return model

def train_one_ticker(ticker, commit_hash):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_ticker(ticker, look_back=LOOK_BACK)

    model = build_model((X_train.shape[1], X_train.shape[2]))

    es = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    y_pred = model.predict(X_test).flatten()

    # Inverse transform
    dummy = np.zeros((len(y_test), 5))
    dummy[:, 0] = y_test
    y_test_inv = scaler.inverse_transform(dummy)[:, 0]

    dummy[:, 0] = y_pred
    y_pred_inv = scaler.inverse_transform(dummy)[:, 0]

    rmse = mean_squared_error(y_test_inv, y_pred_inv) ** 0.5
    mae  = mean_absolute_error(y_test_inv, y_pred_inv)

    out_dir = f"evaluation_output/{commit_hash}"
    os.makedirs(out_dir, exist_ok=True)

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{ticker} Training History")
    plt.legend()
    plt.savefig(f"{out_dir}/training_history_{ticker}.png")
    plt.close()

    # Plot prediction
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_inv, label="Actual (Groundtruth)", color="steelblue")
    plt.plot(y_pred_inv, label="Predicted", color="tomato", linestyle="--")
    plt.title(f"LSTM Forecast — {ticker} | RMSE={rmse:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/prediction_{ticker}.png")
    plt.close()

    # Save model temporarily (can be moved/kept later)
    model.save(f"{out_dir}/model_{ticker}_lstm_baseline.keras")

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
    print("---")
    print(f"rmse_avg:        {rmse_avg:.6f}")
    print(f"training_seconds:{time.time() - t0:.1f}")
    print(f"model:           LSTM baseline look_back={LOOK_BACK}")

    # Save regression report
    report_data = []
    for ticker, (rmse, mae) in results.items():
        report_data.append({"Ticker": ticker, "RMSE": rmse, "MAE": mae})
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(f"{out_dir}/regression_report.csv", index=False)
    try:
        df_report.to_excel(f"{out_dir}/regression_report.xlsx", index=False)
    except Exception as e:
        print("Note: openpyxl might be missing, could not save .xlsx")

    # Generate experiment card
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    card_content = f"""=== EXPERIMENT CARD ===
commit:         {commit_hash}
date:           {date_str}

--- Model ---
type:           LSTM (1 layer baseline)
architecture:   LSTM(64) → Dropout(0.2) → Dense(1)
optimizer:      Adam lr=0.001
epochs_run:     {np.mean(epochs_list):.1f} (avg)

--- Preprocessing ---
scaler:         MinMaxScaler (fit on train only)
look_back:      {LOOK_BACK}
use_log_return: False
features:       close, RSI_14, MA_20, log_return, lag_1

--- Feature Engineering ---
- RSI_14: Relative Strength Index 14 hari
- MA_20: Moving Average 20 hari
- log_return: log(close_t / close_t-1)
- lag_1: close geser 1 hari

--- Results ---
BBCA: rmse={results['BBCA.JK'][0]:.2f}, mae={results['BBCA.JK'][1]:.2f}
BBRI: rmse={results['BBRI.JK'][0]:.2f}, mae={results['BBRI.JK'][1]:.2f}
BMRI: rmse={results['BMRI.JK'][0]:.2f}, mae={results['BMRI.JK'][1]:.2f}
rmse_avg: {rmse_avg:.2f}
status:   keep  |  discard  |  crash
model_files:
  BBCA.JK: model_BBCA.JK_lstm_baseline.keras
  BBRI.JK: model_BBRI.JK_lstm_baseline.keras
  BMRI.JK: model_BMRI.JK_lstm_baseline.keras

--- Why tried ---
Hipotesis: Baseline LSTM(64)+Dropout(0.2), look_back={LOOK_BACK}, baseline features

--- What worked / didn't ---
(to be filled by agent)
"""
    with open(f"{out_dir}/experiment_card.txt", "w") as f:
        f.write(card_content)