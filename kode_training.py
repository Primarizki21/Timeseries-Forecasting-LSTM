import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

def get_data(ticker):
    df = yf.Ticker(ticker).history(period="5y")[['Close']]
    return df.dropna()

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def build_model(input_shape, lr):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def plot_evaluation(ticker, dates, y_full, d_test, y_test, y_pred):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(dates, y_full, label='Harga Historis', color='#1f77b4', alpha=0.6)
    ax1.plot(d_test, y_pred, label='Prediksi Model (Data Test)', color='#ff7f0e', linewidth=2)
    ax1.set_title(f"Tren Harga {ticker}", fontsize=14, fontweight='bold')
    ax2.plot(d_test, y_test, label='Harga Aktual', color='#1f77b4', marker='o', markersize=4)
    ax2.plot(d_test, y_pred, label='Prediksi', color='#ff7f0e', marker='x', markersize=4)
    ax2.set_title(f"Detail Akurasi Prediksi Model pada Data Test", fontsize=12)
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_forecast_only(ticker, forecast_dates, forecast_prices):
    plt.figure(figsize=(8, 4))
    plt.plot(forecast_dates, forecast_prices, label='Prediksi Masa Depan', color='green', marker='s', linestyle='--')
    plt.title(f"Prediksi Harga {ticker} ({len(forecast_prices)} Hari ke Depan)", fontsize=12, fontweight='bold')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel("Harga (Rp)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

def run_pipeline(ticker, seq_length=60, epochs=20, batch_size=32, lr=0.001, forecast_days=5):
    print(f"\n{'='*60}\nMEMPROSES: {ticker}\n{'='*60}")
    
    # 1. Data Prep & Scaling
    df_raw = get_data(ticker)
    df_log_return = np.log(df_raw / df_raw.shift(1)).dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_log_return)
    X, y = create_sequences(scaled_data, seq_length)
    
    # 2. Split 70/15/15
    n = len(X)
    t1, t2 = int(n * 0.7), int(n * 0.85)
    X_train, X_val, X_test = X[:t1], X[t1:t2], X[t2:]
    y_train, y_val, y_test = y[:t1], y[t1:t2], y[t2:]
    dates_test = df_raw.index[seq_length:][t2:]

    # 3. Training
    model = build_model((X_train.shape[1], 1), lr)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)

    # 4. Inverse Transform dari MinMax
    y_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_test_inv = scaler.inverse_transform(y_test)
    
    # 5. Kembalikan ke harga asli dari log return
    price_prev = df_raw.values[seq_length-1:][t2:-1] 
    y_pred_rupiah = price_prev * np.exp(y_pred)
    y_test_rupiah = df_raw.values[seq_length:][t2:] # Ini harga aslinya
    
    # 6. Evaluation Metrics and Plots
    rmse = np.sqrt(mean_squared_error(y_test_rupiah, y_pred_rupiah))
    mae = mean_absolute_error(y_test_rupiah, y_pred_rupiah)
    print(f"Akurasi Model (MAE): Rp {mae:,.2f}")
    print(f"Akurasi Model (RMSE): Rp {rmse:,.2f}")
    plot_evaluation(ticker, df_raw.index[seq_length:], scaler.inverse_transform(y), dates_test, y_test_rupiah, y_pred_rupiah)

    # 7. Multi-day Inference
    last_seq = scaled_data[-seq_length:]
    last_price = df_raw.values[-1]
    preds_scaled = []
    # current_price = last_price
    for _ in range(forecast_days):
        p_scaled = model.predict(last_seq[np.newaxis, :, :], verbose=0)[0]
        p_log = scaler.inverse_transform([p_scaled])[0][0]
        current_price = last_price * np.exp(p_log)
        preds_scaled.append(current_price)
        last_seq = np.append(last_seq[1:], [p_scaled], axis=0)
    
    # forecast_prices = scaler.inverse_transform(preds_scaled)
    forecast_dates = pd.date_range(start=df_raw.index[-1], periods=forecast_days + 1, freq='B')[1:]
    
    # Output Text & Plot Inference
    print(f"Prediksi {forecast_days} Hari Ke Depan untuk {ticker}:")
    for i, price in enumerate(preds_scaled):
        print(f"- {forecast_dates[i].strftime('%Y-%m-%d')}: Rp {price[0]:,.2f}")
    
    plot_forecast_only(ticker, forecast_dates, preds_scaled)

# Eksekusi
# tickers = ['BBCA.JK', 'BMRI.JK', 'BBRI.JK']
tickers = ['BBCA.JK']
for t in tickers:
    run_pipeline(t, seq_length=60, epochs=15, batch_size=16, forecast_days=21)