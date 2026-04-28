# eda_saham.py — EDA untuk BBCA, BBRI, BMRI

import os
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

TICKERS = ["BBCA.JK", "BBRI.JK", "BMRI.JK"]
DATA_DIR = "data_saham"
OUTPUT_DIR = "evaluation_output/eda"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_or_fetch(ticker):
    slug = ticker.replace(".JK", "")
    filepath = os.path.join(DATA_DIR, f"{slug}.csv")

    if os.path.exists(filepath):
        df = pl.read_csv(filepath, try_parse_dates=True)
    else:
        df_pd = yf.download(ticker, period="2y", interval="1d")
        if isinstance(df_pd.columns, pd.MultiIndex):
            df_pd.columns = df_pd.columns.get_level_values(0)
        df_pd = df_pd.reset_index()
        df_pd.columns = [str(c).lower() for c in df_pd.columns]
        df = pl.from_pandas(df_pd)
        df.write_csv(filepath)

    # Ensure date column exists and sorted
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column found for {ticker}")

    df = df.sort(date_col)
    return df, date_col


def add_ma_features(df):
    return df.with_columns([
        pl.col("close").rolling_mean(10).alias("MA_10"),
        pl.col("close").rolling_mean(20).alias("MA_20"),
        pl.col("close").rolling_mean(50).alias("MA_50"),
        pl.col("close").rolling_mean(100).alias("MA_100"),
    ])


def run_decomposition(df, date_col, ticker):
    df_pd = df.select([date_col, "close"]).to_pandas().set_index(date_col)
    df_pd = df_pd.asfreq("B").ffill()

    result = seasonal_decompose(df_pd["close"], model="additive", period=5)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    result.observed.plot(ax=axes[0], title="Observed")
    result.trend.plot(ax=axes[1], title="Trend")
    result.seasonal.plot(ax=axes[2], title="Seasonal")
    result.resid.plot(ax=axes[3], title="Residual")
    fig.suptitle(f"Time Series Decomposition — {ticker}", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"decomposition_{ticker}.png"))
    plt.close()


def run_ma_overlay(df, date_col, ticker):
    df_pd = df.select([date_col, "close", "MA_10", "MA_20", "MA_50", "MA_100"]).to_pandas()

    plt.figure(figsize=(14, 6))
    plt.plot(df_pd[date_col], df_pd["close"], label="Close", color="black", linewidth=1.2)
    plt.plot(df_pd[date_col], df_pd["MA_10"], label="MA_10", alpha=0.8)
    plt.plot(df_pd[date_col], df_pd["MA_20"], label="MA_20", alpha=0.8)
    plt.plot(df_pd[date_col], df_pd["MA_50"], label="MA_50", alpha=0.8)
    plt.plot(df_pd[date_col], df_pd["MA_100"], label="MA_100", alpha=0.8)
    plt.title(f"Harga Closing + Moving Averages — {ticker}")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"ma_overlay_{ticker}.png"))
    plt.close()


def run_ma_subplots(df, date_col, ticker):
    df_pd = df.select([date_col, "close", "MA_10", "MA_20", "MA_50", "MA_100"]).to_pandas()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    mas = [("MA_10", 0, 0), ("MA_20", 0, 1), ("MA_50", 1, 0), ("MA_100", 1, 1)]

    for ma, r, c in mas:
        ax = axes[r][c]
        ax.plot(df_pd[date_col], df_pd["close"], label="Close", color="black", linewidth=1.2)
        ax.plot(df_pd[date_col], df_pd[ma], label=ma, color="tomato")
        ax.set_title(ma)
        ax.legend()

    fig.suptitle(f"Harga Closing + Moving Averages (per panel) — {ticker}", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"ma_subplots_{ticker}.png"))
    plt.close()


def run_adf(df, ticker, adf_results):
    close_series = df.select("close").to_pandas()["close"].dropna()
    stat, pvalue, _, _, critical_values, _ = adfuller(close_series)

    is_stationary = pvalue < 0.05
    conclusion = "STATIONARY" if is_stationary else "NON-STATIONARY"
    recommendation = "" if is_stationary else " => gunakan log_return"

    adf_results.append(f"{ticker}: ADF stat={stat:.4f}, p={pvalue:.4f} => {conclusion}{recommendation}")

    print(f"  {ticker}: ADF stat={stat:.4f}, p={pvalue:.4f} => {conclusion}")


def run_acf_pacf(df, ticker):
    close_series = df.select("close").to_pandas()["close"].dropna()

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(close_series, lags=40, ax=ax)
    ax.set_title(f"ACF — {ticker}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"acf_{ticker}.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_pacf(close_series, lags=40, ax=ax)
    ax.set_title(f"PACF — {ticker}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"pacf_{ticker}.png"))
    plt.close()


if __name__ == "__main__":
    adf_results = []

    for ticker in TICKERS:
        print(f"\n=== Processing {ticker} ===")
        df, date_col = load_or_fetch(ticker)
        df = add_ma_features(df)
        slug = ticker.replace(".JK", "")

        print("  Running decomposition...")
        run_decomposition(df, date_col, slug)

        print("  Running MA overlay...")
        run_ma_overlay(df, date_col, slug)

        print("  Running MA subplots...")
        run_ma_subplots(df, date_col, slug)

        print("  Running ADF test...")
        run_adf(df, slug, adf_results)

        print("  Running ACF/PACF...")
        run_acf_pacf(df, slug)

    # Save ADF results
    with open(os.path.join(OUTPUT_DIR, "adf_results.txt"), "w") as f:
        f.write("\n".join(adf_results) + "\n")

    print("\n=== EDA selesai ===")
    print(f"Hasil tersimpan di: {OUTPUT_DIR}/")
