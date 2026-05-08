# eda_saham.py — EDA untuk BBCA, BBRI, BMRI

import os
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stl._stl import STL
import matplotlib.dates as mdates

# TICKERS = ["BBCA.JK", "BBRI.JK", "BMRI.JK"]
TICKERS = ["BBCA.JK"]
DATA_DIR = "data_saham"
OUTPUT_DIR = "evaluation_output/eda"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_or_fetch(ticker, rentang_waktu):
    slug = ticker.replace(".JK", "")
    filepath = os.path.join(DATA_DIR, f"{slug}.csv")

    if os.path.exists(filepath):
        df = pl.read_csv(filepath, try_parse_dates=True)
    else:
        df_pd = yf.download(ticker, period=rentang_waktu, interval="1d")
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

def run_decomposition_stats(df, date_col, ticker, period=252):
    df_pd = df.select([date_col, "close"]).to_pandas().set_index(date_col)
    df_pd = df_pd.asfreq("B").ffill()

    result = seasonal_decompose(df_pd["close"], model="additive", period=period)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    result.observed.plot(ax=axes[0], title="Observed")
    result.trend.plot(ax=axes[1], title="Trend")
    result.seasonal.plot(ax=axes[2], title="Seasonal")
    result.resid.plot(ax=axes[3], title="Residual")
    fig.suptitle(f"Time Series Decomposition — {ticker}", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"stats_decomposition_{ticker}_{period}.png"))
    plt.close()

# def run_decomposition_stl(df, date_col, ticker, period=63, seasonal=13):
#     df_pd = df.select([date_col, "close"]).to_pandas().set_index(date_col)
#     df_pd = df_pd.asfreq("B").ffill()

#     stl = STL(df_pd["close"], period=period, seasonal=seasonal, robust=True)
#     result = stl.fit()

#     fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
#     axes[0].plot(result.observed);  axes[0].set_title("Observed")
#     axes[1].plot(result.trend);     axes[1].set_title("Trend")
#     axes[2].plot(result.seasonal);  axes[2].set_title("Seasonal (STL)")
#     axes[3].plot(result.resid);     axes[3].set_title("Residual")
#     fig.suptitle(f"STL Decomposition Period {period} Seasonal {seasonal} — {ticker}", fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, f"STL_decomposition_{ticker}_{period}_{seasonal}.png"))
#     plt.close()

def run_decomposition_stl(df, date_col, ticker, period=63, seasonal=13):
    df_pd = df.select([date_col, "close"]).to_pandas().set_index(date_col)
    df_pd = df_pd.asfreq("B").ffill()

    stl = STL(df_pd["close"], period=period, seasonal=seasonal, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    components = [
        (result.observed, "Observed"),
        (result.trend, "Trend"),
        (result.seasonal, "Seasonal (STL)"),
        (result.resid, "Residual")
    ]

    for i, (data, title) in enumerate(components):
        axes[i].plot(data)
        axes[i].set_title(title)
        axes[i].tick_params(labelbottom=True) 
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.suptitle(f"STL Decomposition Period {period} — {ticker}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"STL_decomposition_{ticker}_{period}_{seasonal}.png"))
    plt.close()

def run_ma_overlay(df, date_col, ticker):
    df_pd = df.select([date_col, "close", "MA_10", "MA_20", "MA_50", "MA_100"]).to_pandas()

    plt.figure(figsize=(14, 6))
    plt.plot(df_pd[date_col], df_pd["close"], label="Close", color="black", linewidth=1.2)
    plt.plot(df_pd[date_col], df_pd["MA_10"], label="MA_10", alpha=0.8)
    # plt.plot(df_pd[date_col], df_pd["MA_20"], label="MA_20", alpha=0.8)
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
    # mas = [("MA_10", 0, 0), ("MA_20", 0, 1), ("MA_50", 1, 0), ("MA_100", 1, 1)]
    mas = [("MA_10", 0, 0), ("MA_50", 0, 1), ("MA_100", 1, 0)]

    fig.suptitle(f"Harga Closing + Moving Averages (per panel) — {ticker}", fontweight='bold', fontsize=14)
    for ma, r, c in mas:
        ax = axes[r][c]
        ax.plot(df_pd[date_col], df_pd["close"], label="Close", color="black", linewidth=1.2)
        ax.plot(df_pd[date_col], df_pd[ma], label=ma, color="tomato")
        ax.set_title(ma)
        ax.legend()
    axes[1][1].axis('off')
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

def run_acf_pacf(df, ticker, lags=63):
    close_series = df.select("close").to_pandas()["close"].dropna()
    
    log_return = np.log(close_series).diff().dropna()

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(log_return, lags=lags, ax=ax)
    ax.set_title(f"ACF Log Return — {ticker}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"log_return_acf_{ticker}_lags{lags}.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_pacf(log_return, lags=lags, ax=ax)
    ax.set_title(f"PACF Log Return — {ticker}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"log_return_pacf_{ticker}_lags{lags}.png"))
    plt.close()

def run_seasonality_plot(df, date_col, ticker):
    df_pd = df.select([date_col, "close"]).to_pandas()
    df_pd[date_col] = pd.to_datetime(df_pd[date_col])
    df_pd = df_pd.sort_values(date_col).set_index(date_col)

    # Hitung log return harian
    df_pd["log_return"] = np.log(df_pd["close"] / df_pd["close"].shift(1))
    df_pd = df_pd.dropna()

    df_pd["day_of_week"] = df_pd.index.day_name()
    df_pd["month"]       = df_pd.index.month
    df_pd["month_name"]  = df_pd.index.strftime("%b")
    df_pd["quarter"]     = df_pd.index.quarter.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
    df_pd["year"]        = df_pd.index.year

    DAY_ORDER   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]

    dow_avg  = df_pd.groupby("day_of_week")["log_return"].mean().reindex(DAY_ORDER)
    mom_avg  = df_pd.groupby("month_name")["log_return"].mean().reindex(MONTH_ORDER)
    qoq_avg  = df_pd.groupby("quarter")["log_return"].mean().reindex(QUARTER_ORDER)

    # Heatmap: baris = tahun, kolom = bulan
    heatmap_data = (
        df_pd.groupby(["year", "month"])["log_return"]
        .mean()
        .unstack(level=1)
        .rename(columns={i: MONTH_ORDER[i - 1] for i in range(1, 13)})
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Seasonality Analysis — {ticker}", fontsize=16, fontweight="bold", y=1.01)

    def bar_color(series):
        return ["#2ecc71" if v >= 0 else "#e74c3c" for v in series]

    # --- Panel 1: Day-of-Week ---
    ax1 = axes[0][0]
    ax1.bar(dow_avg.index, dow_avg.values * 100, color=bar_color(dow_avg.values), edgecolor="white")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title("Rata-rata Log Return per Hari", fontweight="bold")
    ax1.set_ylabel("Log Return (%)")
    ax1.set_xlabel("Hari")
    for i, (day, val) in enumerate(dow_avg.items()):
        ax1.text(i, val * 100 + (0.001 if val >= 0 else -0.002),
                 f"{val * 100:.3f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    # --- Panel 2: Month-of-Year ---
    ax2 = axes[0][1]
    ax2.bar(MONTH_ORDER, mom_avg.values * 100, color=bar_color(mom_avg.values), edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Rata-rata Log Return per Bulan", fontweight="bold")
    ax2.set_ylabel("Log Return (%)")
    ax2.set_xlabel("Bulan")
    for i, val in enumerate(mom_avg.values):
        ax2.text(i, val * 100 + (0.001 if val >= 0 else -0.002),
                 f"{val * 100:.3f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)

    # --- Panel 3: Quarter ---
    ax3 = axes[1][0]
    ax3.bar(QUARTER_ORDER, qoq_avg.values * 100, color=bar_color(qoq_avg.values),
            edgecolor="white", width=0.5)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.set_title("Rata-rata Log Return per Kuartal", fontweight="bold")
    ax3.set_ylabel("Log Return (%)")
    ax3.set_xlabel("Kuartal")
    for i, val in enumerate(qoq_avg.values):
        ax3.text(i, val * 100 + (0.001 if val >= 0 else -0.002),
                 f"{val * 100:.3f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

    # --- Panel 4: Heatmap Bulanan per Tahun ---
    ax4 = axes[1][1]
    sns.heatmap(
        heatmap_data * 100,
        ax=ax4,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Log Return (%)"},
    )
    ax4.set_title("Heatmap Log Return Bulanan per Tahun", fontweight="bold")
    ax4.set_xlabel("Bulan")
    ax4.set_ylabel("Tahun")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"seasonality_{ticker}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Seasonality plot tersimpan: seasonality_{ticker}.png")

if __name__ == "__main__":
    adf_results = []

    for ticker in TICKERS:
        print(f"\n=== Processing {ticker} ===")
        df, date_col = load_or_fetch(ticker, '5y')
        df = add_ma_features(df)
        slug = ticker.replace(".JK", "")

        print("  Running Stats decomposition...")
        run_decomposition_stats(df, date_col, slug)

        print("  Running STL decomposition...")
        run_decomposition_stl(df, date_col, slug)

        print("  Running MA overlay...")
        run_ma_overlay(df, date_col, slug)

        print("  Running MA subplots...")
        run_ma_subplots(df, date_col, slug)

        print("  Running ADF test...")
        run_adf(df, slug, adf_results)

        print("  Running ACF/PACF...")
        run_acf_pacf(df, slug)

        print("  Running seasonality plot...")
        run_seasonality_plot(df, date_col, slug)

    # Save ADF results
    with open(os.path.join(OUTPUT_DIR, "adf_results.txt"), "w") as f:
        f.write("\n".join(adf_results) + "\n")

    print("\n=== EDA selesai ===")
    print(f"Hasil tersimpan di: {OUTPUT_DIR}/")
