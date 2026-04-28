# experiment.py (NEW — LSTM baseline)

import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, mean_absolute_error

from prepare import prepare_ticker, TICKERS

LOOK_BACK = 20

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

def train_one_ticker(ticker):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_ticker(ticker)

    model = build_model((X_train.shape[1], X_train.shape[2]))

    es = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    y_pred = model.predict(X_test).flatten()

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae  = mean_absolute_error(y_test, y_pred)

    return rmse, mae


if __name__ == "__main__":
    t0 = time.time()

    results = {}
    rmses = []

    for ticker in TICKERS:
        rmse, mae = train_one_ticker(ticker)
        results[ticker] = (rmse, mae)
        rmses.append(rmse)

        print("---")
        print(f"ticker: {ticker}")
        print(f"rmse: {rmse:.4f}")
        print(f"mae: {mae:.4f}")

    print("---")
    print(f"rmse_avg: {np.mean(rmses):.4f}")
    print(f"training_seconds: {time.time() - t0:.1f}")
    print(f"model: LSTM baseline look_back={LOOK_BACK}")