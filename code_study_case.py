# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sharing Session

# %% [markdown]
# ## Library

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import polars as pl
import nltk
import re
import contractions
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from pandarallel import pandarallel
import squarify
import random
from wordcloud import WordCloud
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

# %%
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"GPU Terdeteksi: {gpu_devices}")
    # Rekomendasi: Aktifkan Memory Growth agar TF tidak langsung "memakan" semua VRAM
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("GPU tidak terdeteksi. Akan menggunakan CPU.")

# %% [markdown]
# ## Housing Data

# %% [markdown]
# ### Load Dataset

# %%
housing_path = 'Housing.csv'
housing_data = pd.read_csv(housing_path)
display(housing_data)

housing_data['area'] = housing_data['area'] * 0.092903
KURS_INR_TO_IDR = 182 
housing_data['price'] = housing_data['price'] * KURS_INR_TO_IDR
display(housing_data.head(5))

# %%
print("Jumlah Missing Values per Kolom:")
print(housing_data.isnull().sum())

print("\nBaris Duplikat:")
print(housing_data[housing_data.duplicated()])


# %% [markdown]
# ### EDA

# %% [markdown]
# #### Bar Chart

# %%
def bar_chart_ok(kolom_eda, nama_label_x):
    df_kol = (
        pl.scan_csv(housing_path)
        .select(pl.col(kolom_eda))
        .group_by(kolom_eda)
        .agg(pl.col(kolom_eda).count().alias('Frekuensi'))
        .sort(kolom_eda, descending=False)
        .collect()
        .to_pandas()
    )
    display(df_kol)

    fig, ax = plt.subplots(figsize=(12,10))
    sns.barplot(data=df_kol, x=kolom_eda, y='Frekuensi', hue=kolom_eda)
    for label in ax.containers:
        ax.bar_label(label, padding=3, fontsize=14, fontweight='bold')
    plt.title(f"Distribusi {nama_label_x}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def bar_chart_subplots(list_kolom_eda, list_nama_label, ncols=2, figsize_per_plot=(12,8)):
    n_plots = len(list_kolom_eda)
    nrows = (n_plots + ncols - 1) // ncols  # Hitung jumlah baris yang dibutuhkan
    
    # Hitung total figsize
    total_width = figsize_per_plot[0] * ncols
    total_height = figsize_per_plot[1] * nrows
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(total_width, total_height))
    
    # Flatten axes array untuk memudahkan iterasi
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for idx, (kolom_eda, nama_label) in enumerate(zip(list_kolom_eda, list_nama_label)):
        # Proses data
        df_kol = (
            pl.scan_csv(housing_path)
            .select(pl.col(kolom_eda))
            .group_by(kolom_eda)
            .agg(pl.col(kolom_eda).count().alias('Frekuensi'))
            .sort(kolom_eda, descending=False)
            .collect()
            .to_pandas()
        )
        
        # Plot di subplot yang sesuai
        sns.barplot(data=df_kol, x=kolom_eda, y='Frekuensi', hue=kolom_eda, ax=axes[idx], palette='husl')
        for label in axes[idx].containers:
            axes[idx].bar_label(label, padding=3, fontsize=14, fontweight='bold')
        axes[idx].set_title(f"Distribusi {nama_label}", fontsize=14, fontweight='bold')
    
    # Sembunyikan axes yang tidak terpakai
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# %%
list_kolom_bar1 = ['bedrooms', 'bathrooms']
list_nama_label1 = ['Jumlah Kamar', 'Jumlah Kamar Mandi']

list_kolom_bar2 = ['stories', 'parking']
list_nama_label2 = ['Tingkat/Lantai Rumah', 'Jumlah Tempat Parkir']
bar_chart_subplots(list_kolom_bar1, list_nama_label1, ncols=2)
bar_chart_subplots(list_kolom_bar2, list_nama_label2, ncols=2)

# %% [markdown]
# #### Box Plot

# %%

# %%
df_ac_garage = housing_data[['price', 'mainroad', 'airconditioning']].copy()

label_map = {
    ('yes', 'yes'): 'AC: Ada\nJalan Utama: Terhubung',
    ('yes', 'no'): 'AC: Ada\nJalan Utama: Tidak Terhubung',
    ('no', 'yes'): 'AC: Tidak Ada\nJalan Utama: Terhubung',
    ('no', 'no'): 'AC: Tidak Ada\nJalan Utama: Tidak Terhubung'
}

df_ac_garage['Kategori'] = df_ac_garage.apply(
    lambda row: label_map[(row['airconditioning'], row['mainroad'])], axis=1
)

medians = df_ac_garage.groupby('Kategori')['price'].median().sort_values(ascending=False)
kategori_urut = medians.index.tolist()

plt.figure(figsize=(12, 8))
ax = sns.boxplot(data=df_ac_garage, x='Kategori', y='price',  order=kategori_urut, palette='Set2')

plt.title('Distribusi Harga berdasarkan AC dan Jalan Utama', fontsize=14, fontweight='bold')

def format_rupiah(x, pos=None):
    if x >= 1_000_000_000:  # Miliar (≥ 1M)
        return f'Rp {x/1_000_000_000:.1f}Miliar'
    elif x >= 1_000_000:    # Juta (≥ 1jt)
        return f'Rp {x/1_000_000:.1f}Juta'
    elif x >= 1_000:        # Ribuan
        return f'Rp {x/1_000:.0f}Ribu'
    else:
        return f'Rp {int(x):,}'

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_rupiah))

for i, kategori in enumerate(kategori_urut):
    m = medians[kategori]
    ax.text(i, m, format_rupiah(m), ha='center', va='bottom', 
            fontweight='bold', fontsize=9)

plt.xlabel('Kategori')
plt.ylabel('Harga')
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# %%
df_furnished = housing_data[['price', 'furnishingstatus']]

# %% [markdown]
# #### Scatter Plot

# %%
df_scatter = housing_data.copy()
median_luas = df_scatter['area'].median()
median_harga = df_scatter['price'].median()

def tentukan_kuadran(row):
    if row['area'] >= median_luas and row['price'] >= median_harga:
        return 'Luas & Mahal'
    elif row['area'] < median_luas and row['price'] >= median_harga:
        return 'Kecil & Mahal'
    elif row['area'] < median_luas and row['price'] < median_harga:
        return 'Kecil & Murah'
    else:
        return 'Luas & Murah'

df_scatter['Kuadran'] = df_scatter.apply(tentukan_kuadran, axis=1)
fig, ax = plt.subplots(figsize=(12, 10))
plt.fill_between(x=[df_scatter['area'].min(), median_luas], 
                 y1=median_harga, y2=df_scatter['price'].max(),
                 color='orange', alpha=0.15)
plt.fill_between(x=[median_luas, df_scatter['area'].max()], 
                 y1=median_harga, y2=df_scatter['price'].max(),
                 color='red', alpha=0.15)
plt.fill_between(x=[df_scatter['area'].min(), median_luas], 
                 y1=df_scatter['price'].min(), y2=median_harga,
                 color='blue', alpha=0.15)
plt.fill_between(x=[median_luas, df_scatter['area'].max()], 
                 y1=df_scatter['price'].min(), y2=median_harga,
                 color='green', alpha=0.15)

plt.axhline(y=median_harga, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
plt.axvline(x=median_luas, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

colors_furnishing = {
    'furnished': '#FF6B6B',      
    'semi-furnished': '#FFA500', 
    'unfurnished': '#4ECDC4'     
}
sns.scatterplot(data=df_scatter, x='area', y='price', hue='furnishingstatus', 
                palette=colors_furnishing, s=100, edgecolor='white', ax=ax)

# Format sumbu y untuk harga
def format_rupiah(x, pos=None):
    if x >= 1_000_000_000:  # Miliar (≥ 1M)
        return f'Rp {x/1_000_000_000:.1f}Miliar'
    elif x >= 1_000_000:    # Juta (≥ 1jt)
        return f'Rp {x/1_000_000:.1f}Juta'
    elif x >= 1_000:        # Ribuan
        return f'Rp {x/1_000:.0f}Ribu'
    else:
        return f'Rp {int(x):,}'

ax.yaxis.set_major_formatter(FuncFormatter(format_rupiah))
plt.xlabel('Luas Tanah (sq ft)', fontsize=12)
plt.ylabel('Harga Rumah', fontsize=12)
plt.title('Analisis Kuadran: Luas Tanah vs Harga Rumah', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(title='Status Perabotan', bbox_to_anchor=(1, 1), loc='upper right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Preprocessing

# %%
cols_housing = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
prepro_housing = housing_data.copy()
prepro_housing[cols_housing] = prepro_housing[cols_housing].apply(lambda x: x.map({'yes':1, 'no':0}))
display(prepro_housing[cols_housing])

# %%
display(prepro_housing['furnishingstatus'].value_counts())

# %%
prepro1_housing = prepro_housing.copy()
prepro1_housing = pd.get_dummies(prepro1_housing, columns=['furnishingstatus'], dtype=int)
display(prepro1_housing)

# %% [markdown]
# ### EDA lama

# %%
# --- 3a. Distribusi Target Variable (Price) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(prepro1_housing['price'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Distribusi Price (Original)')
axes[0].set_xlabel('Price')
axes[1].hist(np.log1p(prepro1_housing['price']), bins=30, color='salmon', edgecolor='white')
axes[1].set_title('Distribusi Log(Price)')
axes[1].set_xlabel('Log Price')
plt.tight_layout()
plt.savefig('01_price_distribution.png', dpi=100)
plt.show()

# %%
# --- 3b. Correlation Heatmap ---
plt.figure(figsize=(11, 8))
corr = prepro1_housing.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=100)
plt.show()

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Data
X = prepro1_housing.drop('price', axis=1)
y = prepro1_housing['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== SCALING =====
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ===== LINEAR REGRESSION =====
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred_train = model.predict(X_train_scaled)
y_pred_test  = model.predict(X_test_scaled)

# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred_train = model.predict(X_train)
# y_pred_test  = model.predict(X_test)

# ===== EVALUASI =====
for split, y_true, y_pred in [('Train', y_train, y_pred_train), ('Test', y_test, y_pred_test)]:
    print(f"📊 {split}")
    print(f"   R²   : {r2_score(y_true, y_pred):.4f}")
    print(f"   MAE  : Rp {mean_absolute_error(y_true, y_pred):,.2f}")
    print(f"   MSE  : Rp {mean_squared_error(y_true, y_pred):,.2f}")
    print(f"   RMSE : Rp {np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}\n")

# ===== PERSAMAAN REGRESI =====
def print_equation(label, intercept, coefs, feature_names):
    eq = f"price = {intercept:,.2f}"
    for coef, name in zip(coefs, feature_names):
        eq += f" {'+'if coef>=0 else'-'} {abs(coef):,.2f}·{name}"
    print(f"📐 {label}\n{eq}\n")

# Skala MinMax
print_equation("Skala MinMax (0–1)", model.intercept_, model.coef_, X.columns)

# Skala asli
ranges    = scaler.data_max_ - scaler.data_min_
coef_ori  = model.coef_ / ranges
inter_ori = model.intercept_ - np.sum(coef_ori * scaler.data_min_)
print_equation("Skala Asli", inter_ori, coef_ori, X.columns)

# ===== FEATURE IMPORTANCE PLOT =====
fi = pd.DataFrame({'Fitur': X.columns, 'Koefisien': model.coef_}) \
       .sort_values('Koefisien', key=abs, ascending=True)

colors = ['#FF6B6B' if c > 0 else '#4ECDC4' for c in fi['Koefisien']]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(fi['Fitur'], fi['Koefisien'], color=colors, height=0.6)
# ax.bar_label(bars, labels=[f'{v/1e9:.1f}M' for v in fi['Koefisien']], 
#              padding=4, fontsize=9)
ax.axvline(0, color='#333', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Koefisien')
ax.set_title('Fitur yang Berpengaruh', fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.show()

# %%
display(X_train.head(5))
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
display(X_train_scaled_df.head(5))

# %% [markdown]
# ## Time-Series Forecasting Tesla

# %% [markdown]
# ### Load Dataset

# %%
df_bca_5 = pd.read_csv('data_saham/BBCA.csv')
df_bca_5['log_return'] = np.log(df_bca_5['close'] / df_bca_5['close'].shift(1))

# Menampilkan hasil
display(df_bca_5[['close', 'log_return']].head(10))

# %%
# ambil data 2 tahun terakhir
end = datetime.now()
start = datetime(end.year - 2, end.month, end.day)

data_tsla = yf.download('TSLA', start=start, end=end)
data_tsla

# %% [markdown]
# ### Closing Price

# %%
plt.figure(figsize=(12,16))
data_tsla['Close'].plot()
plt.ylabel('Close')
plt.xlabel(None)
plt.title(f'Closing Price Tesla')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Moving Average

# %%
ma_day = [10, 20, 50]

for ma in ma_day:
    column_name = f"MA for {ma} days"
    data_tsla[column_name] = data_tsla['Close'].rolling(ma).mean()

data_tsla[['Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot()
plt.title('Moving Average Tesla')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Training Model LSTM

# %%
# 1. AMBIL DATA
ticker = 'TSLA'
# Menggunakan history agar tidak pusing dengan MultiIndex
df = yf.Ticker(ticker).history(period='5y') 
data = df.filter(['Close'])
dataset = data.values

# 2. PREPROCESSING
# Scaling data ke rentang 0-1 (Penting untuk kestabilan LSTM)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Menentukan training length (80% data)
training_data_len = int(np.ceil(len(dataset) * .8))

# Membuat data training dengan "Look-back Window" (misal 60 hari)
# Artinya: Model melihat 60 hari ke belakang untuk menebak hari ke-61
train_data = scaled_data[0:int(training_data_len), :]
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape data (LSTM butuh input 3D: [samples, time steps, features])
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 3. BUILD MODEL LSTM
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2), # Mencegah overfitting
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1) # Output harga prediksi
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 4. TRAINING
# Epochs: berapa kali model belajar, Batch Size: jumlah data per iterasi
model.fit(x_train, y_train, batch_size=32, epochs=5)

# 5. PREPARING TEST DATA
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :] # Harga asli untuk evaluasi

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 6. EVALUASI & PREDIKSI
predictions = model.predict(x_test)
# Kembalikan ke angka harga asli (Inverse Scaling)
predictions = scaler.inverse_transform(predictions)

# Hitung error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f'RMSE: {rmse}')

# 7. OUTPUT PLOT
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(12,6))
plt.title(f'Prediksi Harga Saham {ticker} dengan LSTM')
plt.xlabel('Tanggal')
plt.ylabel('Harga Close (USD)')
plt.plot(train['Close'], label='Data Training')
plt.plot(valid['Close'], label='Harga Asli (Ground Truth)')
plt.plot(valid['Predictions'], label='Hasil Prediksi')
plt.legend(loc='lower right')
plt.show()

# %% [markdown]
# ## Time-Series Forecasting Bank

# %% [markdown]
# ### Load Dataset

# %%
# tickers = ['BBCA.JK', 'BMRI.JK', 'BBRI.JK']

df_bca = yf.Ticker('BBCA.JK').history(period="2y")
df_mandiri = yf.Ticker('BBCA.JK').history(period="2y")
df_bri = yf.Ticker('BBCA.JK').history(period="2y")

display(df_bca.head(5))
display(df_mandiri.head(5))
display(df_bri.head(5))


# %% [markdown]
# ### Line Chart

# %% [markdown]
# ### Training 1

# %%
def get_data(ticker):
    df = yf.Ticker(ticker).history(period="2y")[['Close']]
    df.dropna(inplace=True)
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def split_data(X, y, dates):
    n = len(X)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)

    X_train, y_train = X[:t1], y[:t1]
    X_val, y_val = X[t1:t2], y[t1:t2]
    X_test, y_test = X[t2:], y[t2:]
    
    d_train = dates[:t1]
    d_val = dates[t1:t2]
    d_test = dates[t2:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, d_train, d_val, d_test

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

def format_plot(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)

def plot_results(ticker, dates_full, y_full, dates_test, y_test, y_pred):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates_full, y_full, label='Harga Aktual (Full)', color='#1f77b4', linewidth=1.5)
    ax.plot(dates_test, y_pred, label='Hasil Prediksi Model', color='#ff7f0e', linewidth=2)
    ax.set_title(f"{ticker} - Harga Aktual vs Prediksi Model", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Harga (Rp)", fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    format_plot(ax)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates_test, y_test, label='Harga Aktual (Data Test)', color='#1f77b4', marker='o', markersize=4)
    ax.plot(dates_test, y_pred, label='Hasil Prediksi Model', color='#ff7f0e', marker='x', markersize=4)
    ax.set_title(f"{ticker} - Detail Prediksi pada Data Test", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Harga (Rp)", fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    format_plot(ax)
    plt.tight_layout()
    plt.show()

def forecast_next_week(model, last_seq, scaler, days=5):
    forecast = []
    curr_seq = last_seq.copy()
    
    for _ in range(days):
        pred = model.predict(curr_seq[np.newaxis, :, :], verbose=0)[0]
        forecast.append(pred)
        curr_seq = np.append(curr_seq[1:], [pred], axis=0)
        
    return scaler.inverse_transform(forecast)

def run_pipeline(ticker, seq_length=60):
    print(f"\n{'='*50}\nMEMPROSES DATA: {ticker}\n{'='*50}")
    df = get_data(ticker)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    dates = df.index[seq_length:]
    X, y = create_sequences(scaled_data, seq_length)
    
    X_train, y_train, X_val, y_val, X_test, y_test, _, _, d_test = split_data(X, y, dates)
    
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=0)
    
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_inv = scaler.inverse_transform(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
    mae = mean_absolute_error(y_test_inv, y_pred)
    
    print(f"Hasil Evaluasi Model untuk {ticker}:")
    print(f"- RMSE : Rp {rmse:,.2f} (Estimasi besaran error rata-rata)")
    print(f"- MAE  : Rp {mae:,.2f} (Simpangan rata-rata absolut dari harga asli)\n")
    
    y_full_inv = scaler.inverse_transform(y)
    plot_results(ticker, dates, y_full_inv, d_test, y_test_inv, y_pred)
    
    last_seq = scaled_data[-seq_length:]
    next_week = forecast_next_week(model, last_seq, scaler)
    
    print(f"Prediksi 1 Minggu Kedepan (5 Hari Bursa) untuk {ticker}:")
    for i, price in enumerate(next_week):
        print(f"- Hari ke-{i+1}: Rp {price[0]:,.2f}")

tickers = ['BBCA.JK', 'BMRI.JK', 'BBRI.JK']
for t in tickers:
    run_pipeline(t)


# %% [markdown]
# ### Training 2

# %%
def get_data(ticker):
    df = yf.Ticker(ticker).history(period="2y")[['Close']]
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
    
    # Plot 1: Full History + Prediction Overlay
    ax1.plot(dates, y_full, label='Harga Historis', color='#1f77b4', alpha=0.6)
    ax1.plot(d_test, y_pred, label='Prediksi pada Data Test', color='#ff7f0e', linewidth=2)
    ax1.set_title(f"Tren Harga {ticker} (2 Tahun Terakhir)", fontsize=14, fontweight='bold')
    
    # Plot 2: Zoom-in Test Data
    ax2.plot(d_test, y_test, label='Harga Aktual', color='#1f77b4', marker='o', markersize=4)
    ax2.plot(d_test, y_pred, label='Prediksi', color='#ff7f0e', marker='x', markersize=4)
    ax2.set_title(f"Detail Akurasi Model pada Data Test", fontsize=12)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_forecast(ticker, last_dates, last_prices, forecast_dates, forecast_prices):
    plt.figure(figsize=(10, 5))
    plt.plot(last_dates, last_prices, label='Harga Terakhir (Historis)', color='black', marker='o')
    plt.plot(forecast_dates, forecast_prices, label='Proyeksi 1 Minggu Ke Depan', color='green', linestyle='--', marker='s')
    
    plt.title(f"Proyeksi Harga {ticker} untuk 5 Hari Bursa Mendatang", fontsize=14, color='green', fontweight='bold')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel("Harga (Rp)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

def run_pipeline(ticker, seq_length=60, epochs=20, batch_size=32, lr=0.001):
    print(f"\n{'='*60}\nRUNNING MODEL FOR: {ticker} (Lookback: {seq_length}, Epochs: {epochs})\n{'='*60}")
    
    # 1. Data Prep
    df = get_data(ticker)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X, y = create_sequences(scaled_data, seq_length)
    
    # 2. Split 70/15/15
    n = len(X)
    t1, t2 = int(n * 0.7), int(n * 0.85)
    X_train, X_val, X_test = X[:t1], X[t1:t2], X[t2:]
    y_train, y_val, y_test = y[:t1], y[t1:t2], y[t2:]
    dates_test = df.index[seq_length:][t2:]

    # 3. Training
    model = build_model((X_train.shape[1], 1), lr)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)

    # 4. Evaluation
    y_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_test_inv = scaler.inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
    print(f"ERROR RATA-RATA (RMSE): Rp {rmse:,.2f}")

    plot_evaluation(ticker, df.index[seq_length:], scaler.inverse_transform(y), dates_test, y_test_inv, y_pred)

    # 5. Inference 1 Week (5 Days)
    last_seq = scaled_data[-seq_length:]
    preds_scaled = []
    for _ in range(5):
        p = model.predict(last_seq[np.newaxis, :, :], verbose=0)[0]
        preds_scaled.append(p)
        last_seq = np.append(last_seq[1:], [p], axis=0)
    
    forecast_prices = scaler.inverse_transform(preds_scaled)
    forecast_dates = pd.date_range(start=df.index[-1], periods=6, freq='B')[1:]
    
    # Plot forecast bersama 10 hari terakhir agar terlihat trennya
    plot_forecast(ticker, df.index[-10:], df['Close'].values[-10:], forecast_dates, forecast_prices)

# Contoh penggunaan dengan parameter yang bisa diubah-ubah
tickers = ['BBCA.JK', 'BMRI.JK', 'BBRI.JK']
for t in tickers:
    run_pipeline(t, seq_length=30, epochs=15, batch_size=16, lr=0.001)


# %% [markdown]
# ### Training 3

# %%
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
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_raw)
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

    # 4. Evaluation Plots
    y_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_test_inv = scaler.inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
    mae = mean_absolute_error(y_test_inv, y_pred)
    print(f"Akurasi Model (MAE): Rp {mae:,.2f}")
    print(f"Akurasi Model (RMSE): Rp {rmse:,.2f}")
    plot_evaluation(ticker, df_raw.index[seq_length:], scaler.inverse_transform(y), dates_test, y_test_inv, y_pred)

    # 5. Multi-day Inference
    last_seq = scaled_data[-seq_length:]
    preds_scaled = []
    for _ in range(forecast_days):
        p = model.predict(last_seq[np.newaxis, :, :], verbose=0)[0]
        preds_scaled.append(p)
        last_seq = np.append(last_seq[1:], [p], axis=0)
    
    forecast_prices = scaler.inverse_transform(preds_scaled)
    forecast_dates = pd.date_range(start=df_raw.index[-1], periods=forecast_days + 1, freq='B')[1:]
    
    # Output Text & Plot Inference
    print(f"Prediksi {forecast_days} Hari Ke Depan untuk {ticker}:")
    for i, price in enumerate(forecast_prices):
        print(f"- {forecast_dates[i].strftime('%Y-%m-%d')}: Rp {price[0]:,.2f}")
    
    plot_forecast_only(ticker, forecast_dates, forecast_prices)

# Eksekusi
# tickers = ['BBCA.JK', 'BMRI.JK', 'BBRI.JK']
tickers = ['BBCA.JK']
for t in tickers:
    run_pipeline(t, seq_length=60, epochs=15, batch_size=16, forecast_days=21)


# %% [markdown]
# ### Training 4

# %%
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
    ax2.set_title(f"Detail Akurasi Prediksi Model pada Data Test", fontsize=14, fontweight='bold')
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

    # 3. Training
    model = build_model((X_train.shape[1], 1), lr)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)

    # 4. Inverse Transform dari MinMax
    y_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    
    # 5. Kembalikan ke harga asli dari log return
    price_prev = df_raw.values[seq_length-1:][t2:t2 + len(y_pred)]
    y_pred_rupiah = price_prev * np.exp(y_pred)
    y_test_rupiah = df_raw.values[seq_length:][t2:t2 + len(y_pred)]
    dates_test = df_raw.index[seq_length:][t2:t2 + len(y_pred_rupiah)]
    
    # 6. Evaluation Metrics and Plots
    rmse = np.sqrt(mean_squared_error(y_test_rupiah, y_pred_rupiah))
    mae = mean_absolute_error(y_test_rupiah, y_pred_rupiah)
    print(f"Akurasi Model (MAE): Rp {mae:,.2f}")
    print(f"Akurasi Model (RMSE): Rp {rmse:,.2f}")
    plot_evaluation(ticker, df_raw.index[seq_length+1:], df_raw.values[seq_length+1:], dates_test, y_test_rupiah, y_pred_rupiah)

    # 7. Multi-day Inference
    last_seq = scaled_data[-seq_length:]
    last_price = df_raw.values[-1]
    preds_scaled = []
    for _ in range(forecast_days):
        p_scaled = model.predict(last_seq[np.newaxis, :, :], verbose=0)[0]
        p_log = scaler.inverse_transform([p_scaled])[0][0]
        current_price = last_price * np.exp(p_log)
        preds_scaled.append(current_price)
        last_seq = np.append(last_seq[1:], [p_scaled], axis=0)
    
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
    run_pipeline(t, seq_length=50, epochs=15, batch_size=16, forecast_days=21)

# %% [markdown]
# ## Sentiment Analysis

# %% [markdown]
# Analisis Headline Berita

# %%
path_headline = "all-data.csv"
head_data = pl.read_csv(
    path_headline, 
    has_header=False
).to_pandas()
display(head_data)

# %% [markdown]
# ### Contoh Data

# %%
# print(head_data[head_data['column_1'] == 'positive'].iloc[0]['column_2'])
# print(head_data[head_data['column_1'] == 'negative'].iloc[5]['column_2'])
print(head_data[head_data['column_1'] == 'neutral'].iloc[50]['column_2'])


# %%
def plot_word_frequency(df, column, top_n=20, figsize=(14, 10), palette='viridis', title=None):
    all_words = ' '.join(df[column].dropna()).split()
    word_freq = Counter(all_words)
    top_words = pd.DataFrame(word_freq.most_common(top_n), columns=['word', 'count'])

    plot_title = title or f'Top {top_n} Kata Paling Sering Muncul'

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=top_words, x='count', y='word', hue='word',palette=palette, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, padding=4, fontsize=14, fontweight='bold')
    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.xlabel('Frequency', fontsize=14, fontweight='bold')
    plt.ylabel('Word')
    plt.tight_layout()
    plt.savefig(f'word_freq_{column}.png', dpi=150)
    plt.show()

    return word_freq


# %% [markdown]
# ### EDA Before Prepro

# %%
raw_data = head_data.copy()
raw_data_plot = plot_word_frequency(raw_data, top_n=10, column='column_2')

# %% [markdown]
# ### Rata-Rata Panjang Karakter per Sentimen dan Distribusi Label

# %%
# labeL_jumlah = [jumlah for jumlah in len_karakter['char_count_per_label']]

# fig, ax = plt.subplots(figsize=(14,10))
# sns.barplot(data=len_karakter, x='column_1', y='char_count_per_label', hue='column_1', ax=ax)
# for i, container in enumerate(ax.containers):
#     ax.bar_label(container, labels=[labeL_jumlah[i]], fontsize=11, fontweight='bold')

# plt.title("Distribusi Panjang karakter per Sentimen Berita")
# plt.xlabel('Sentimen')
# plt.ylabel('Panjang Karakter')
# plt.show()

# %%
# labeL_jumlah = [jumlah for jumlah in distribusi_label['count']]

# fig, ax = plt.subplots(figsize=(14,10))
# sns.barplot(data=distribusi_label, x='column_1', y='count', hue='column_1', ax=ax)
# for i, container in enumerate(ax.containers):
#     ax.bar_label(container, labels=[labeL_jumlah[i]], fontsize=11, fontweight='bold')

# plt.title("Distribusi Label Sentimen Berita")
# plt.xlabel('Sentimen')
# plt.ylabel('Frekuensi')
# plt.show()

# %%
distribusi_label = head_data['column_1'].value_counts().reset_index()
distribusi_label = distribusi_label.sort_values('column_1', ascending=False)
display(distribusi_label)

mean_karakter = (
    pl.from_pandas(head_data)
    .with_columns(
        pl.col("column_2").str.strip_chars().str.len_chars().alias("char_count")
    )
    .group_by('column_1')
    .agg(
        pl.col('char_count').mean().alias('char_count_per_label')
    )
    .sort('column_1', descending=True)
    .to_pandas()
)
display(mean_karakter)

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# Chart 1: Distribusi Jumlah Sentimen
sns.barplot(data=distribusi_label, x='column_1', y='count', hue='column_1', palette='viridis', ax=ax[0])
ax[0].set_title("Distribusi Frekuensi Sentimen Berita", fontsize=14, pad=20, fontweight='bold')
ax[0].set_xlabel("Sentimen", fontsize=12)
ax[0].set_ylabel("Jumlah Berita", fontsize=12)

# Menggunakan bar_label untuk Chart 1
# ax[0].containers menyimpan objek bar yang dibuat oleh seaborn
for container in ax[0].containers:
    ax[0].bar_label(container, padding=3, fontsize=14, fontweight='bold')

# Chart 2: Distribusi Rata-rata Panjang Karakter
sns.barplot(data=mean_karakter, x='column_1', y='char_count_per_label', hue='column_1', palette='magma', ax=ax[1])
ax[1].set_title("Rata-rata Panjang Karakter per Sentimen", fontsize=14, pad=20, fontweight='bold')
ax[1].set_xlabel("Sentimen", fontsize=12)
ax[1].set_ylabel("Panjang Karakter", fontsize=12)

# Menggunakan bar_label untuk Chart 2
# fmt='%.0f' digunakan untuk membulatkan nilai (seperti fungsi int() sebelumnya)
for container in ax[1].containers:
    ax[1].bar_label(container, padding=3, fmt='%.0f', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### KDE PLot

# %%
distribusi_karakter = (
    pl.from_pandas(head_data)
    .with_columns(
        pl.col("column_2").str.strip_chars().str.len_chars().alias("char_count")
    )
    .to_pandas()
)

# Asumsi df adalah dataframe kamu yang berisi kolom 'sentiment' dan 'char_count'
plt.figure(figsize=(14, 10))

# Menggunakan sns.kdeplot
sns.kdeplot(
    data=distribusi_karakter, 
    x='char_count', 
    hue='column_1', 
    fill=True,            # Memberi warna di bawah kurva
    common_norm=False,    # KUNCI UTAMA: Agar tiap label punya skala density-nya sendiri
    palette='viridis', 
    alpha=.5,             # Transparansi agar area yang bertumpuk terlihat
    linewidth=2
)

plt.title("Analisis Distribusi Panjang Karakter Berdasarkan Sentimen", fontsize=14, fontweight='bold')
plt.xlabel("Panjang Karakter", fontsize=12)
plt.ylabel("Density (Probabilitas)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %%
# Set style agar terlihat bersih
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

# Membuat Histogram
sns.histplot(
    data=distribusi_karakter, 
    x='char_count', 
    hue='column_1',        # Membedakan warna berdasarkan sentimen
    stat='density',        # PENTING: Menggunakan densitas agar skala Y sebanding
    common_norm=False,     # PENTING: Menormalkan setiap label secara mandiri
    kde=True,              # Menampilkan garis KDE di atas histogram
    palette='viridis',     # Warna yang kontras dan profesional
    alpha=0.3,             # Transparansi agar area tumpang tindih terlihat
    element='step',        # Gaya 'step' membuat batas antar bar lebih jelas
    bins=30                # Jumlah kotak, bisa disesuaikan (makin besar makin detail)
)

# Menambah label dan judul
plt.title("Histogram Distribusi Panjang Karakter per Sentimen (Normalized)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Panjang Karakter", fontsize=12)
plt.ylabel("Density (Probabilitas)", fontsize=12)

# Memberi keterangan tambahan
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Sentiment Lexicon Analysis

# %%
tes_lm = pl.read_csv('Loughran-McDonald_MasterDictionary_1993-2025.csv').to_pandas()
display(tes_lm.head(5))
display(tes_lm[tes_lm['Negative'] > 0].head(5))
display(tes_lm[tes_lm['Positive'] > 0].head(5))

# %%
# 1. Load Loughran-McDonald
lm = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2025.csv')

# Ambil kata positif dan negatif saja
lm_positive = set(lm[lm['Positive'] > 0]['Word'].str.lower())
lm_negative = set(lm[lm['Negative'] > 0]['Word'].str.lower())

# 2. Fungsi hitung skor lexicon per headline
def score_lexicon(text):
    tokens = str(text).lower().split()
    pos = sum(1 for t in tokens if t in lm_positive)
    neg = sum(1 for t in tokens if t in lm_negative)
    return pos, neg

# 3. Terapkan ke dataframe — pakai kolom RAW (hanya lowercase)
lexicon_data = head_data.copy()
lexicon_data['head_lower'] = lexicon_data['column_2'].str.lower()
lexicon_data[['lm_pos', 'lm_neg']] = lexicon_data['head_lower'].apply(
    lambda x: pd.Series(score_lexicon(x))
)
lexicon_data['lm_sentiment'] = lexicon_data.apply(
    lambda r: 'positive' if r['lm_pos'] > r['lm_neg']
              else ('negative' if r['lm_neg'] > r['lm_pos'] else 'neutral'),
    axis=1
)

# 4. PLOT — 2 baris 2 kolom
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Loughran-McDonald Lexicon Analysis — Financial Headlines", 
             fontsize=16, fontweight='bold')

ax1, ax2, ax3, ax4 = axes.flatten()

# Plot 1 — Rata-rata skor positif & negatif per kelas label asli
avg_scores = lexicon_data.groupby('column_1')[['lm_pos', 'lm_neg']].mean().reset_index()
avg_melted = avg_scores.melt(id_vars='column_1', 
                              value_vars=['lm_pos', 'lm_neg'],
                              var_name='score_type', value_name='avg_score')
avg_melted['score_type'] = avg_melted['score_type'].map(
    {'lm_pos': 'Positive Words', 'lm_neg': 'Negative Words'}
)
sns.barplot(data=avg_melted, x='column_1', y='avg_score', 
            hue='score_type', ax=ax1, palette=['steelblue', 'salmon'])
ax1.set_title('Rata-rata Kata Positif & Negatif per Kelas', fontweight='bold')
ax1.set_xlabel('Label Sentimen Asli')
ax1.set_ylabel('Rata-rata Jumlah Kata')
ax1.spines[['top', 'right']].set_visible(False)

# Plot 2 — Confusion: label asli vs label lexicon
confusion = pd.crosstab(lexicon_data['column_1'], lexicon_data['lm_sentiment'], normalize='index') * 100
sns.heatmap(confusion, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2,
            cbar_kws={'label': '%'})
ax2.set_title('Label Asli vs Prediksi Lexicon (%)', fontweight='bold')
ax2.set_xlabel('Prediksi Lexicon')
ax2.set_ylabel('Label Asli')

# Plot 3 — KDE distribusi lm_neg per kelas
for sentiment in ['positive', 'negative', 'neutral']:
    subset = lexicon_data[lexicon_data['column_1'] == sentiment]['lm_neg']
    sns.kdeplot(subset, label=sentiment, fill=True, alpha=0.3, ax=ax3)
ax3.set_title('Distribusi Kata Negatif (LM) per Kelas', fontweight='bold')
ax3.set_xlabel('Jumlah Kata Negatif')
ax3.set_ylabel('Densitas')
ax3.legend()
ax3.spines[['top', 'right']].set_visible(False)

# Plot 4 — KDE distribusi lm_pos per kelas
for sentiment in ['positive', 'negative', 'neutral']:
    subset = lexicon_data[lexicon_data['column_1'] == sentiment]['lm_pos']
    sns.kdeplot(subset, label=sentiment, fill=True, alpha=0.3, ax=ax4)
ax4.set_title('Distribusi Kata Positif (LM) per Kelas', fontweight='bold')
ax4.set_xlabel('Jumlah Kata Positif')
ax4.set_ylabel('Densitas')
ax4.legend()
ax4.spines[['top', 'right']].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('lm_lexicon_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# 1. Load Loughran-McDonald
lm = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2025.csv')
lm_positive = set(lm[lm['Positive'] > 0]['Word'].str.lower())
lm_negative = set(lm[lm['Negative'] > 0]['Word'].str.lower())

# 2. Fungsi hitung skor lexicon
def score_lexicon(text):
    tokens = str(text).lower().split()
    pos = sum(1 for t in tokens if t in lm_positive)
    neg = sum(1 for t in tokens if t in lm_negative)
    return pos, neg

# 3. Terapkan ke dataframe
lexicon_data = head_data.copy()
lexicon_data['head_lower'] = lexicon_data['column_2'].str.lower()
lexicon_data[['lm_pos', 'lm_neg']] = lexicon_data['head_lower'].apply(
    lambda x: pd.Series(score_lexicon(x))
)

# 4. Hitung top kata per sentimen
top_n = 10  # ganti 20 kalau mau lebih banyak

def get_top_lm_words(lexicon_data, label, lexicon_set, top_n):
    texts = lexicon_data[lexicon_data['column_1'] == label]['head_lower']
    all_tokens = ' '.join(texts).split()
    matched = [t for t in all_tokens if t in lexicon_set]
    return pd.DataFrame(Counter(matched).most_common(top_n), 
                        columns=['word', 'count'])

# PLOT — 1 baris 2 kolom
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Analisis Kosakata per Sentimen (Kamus Loughran-McDonald)",
             fontsize=16, fontweight='bold')

# Kiri — Rata-rata skor per kelas
avg_scores = lexicon_data.groupby('column_1')[['lm_pos', 'lm_neg']].mean().reset_index()
avg_melted = avg_scores.melt(id_vars='column_1',
                              value_vars=['lm_pos', 'lm_neg'],
                              var_name='Jenis Kata', value_name='avg_score')
avg_melted['Jenis Kata'] = avg_melted['Jenis Kata'].map(
    {'lm_pos': 'Kata Positif', 'lm_neg': 'Kata Negatif'}
)

# Kiri — tambah label di bar rata-rata skor
sns.barplot(data=avg_melted, x='column_1', y='avg_score',
            hue='Jenis Kata', ax=ax1, palette=['steelblue', 'salmon'])
ax1.set_title('Perbandingan Bobot Kata Positif vs Negatif per Sentimen Berita', fontweight='bold')
ax1.set_xlabel('Label Sentimen')
ax1.set_ylabel('Rata-rata Jumlah Kata')
ax1.spines[['top', 'right']].set_visible(False)
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.2f', padding=3, fontsize=12, fontweight='bold')

# Kanan — Top kata LM yang muncul di seluruh dataset
all_pos_words = get_top_lm_words(lexicon_data, 'positive', lm_positive, top_n)
all_neg_words = get_top_lm_words(lexicon_data, 'negative', lm_negative, top_n)

# Gabung jadi satu dataframe untuk grouped bar
all_pos_words['Jenis Kata'] = 'Kata Positif'
all_neg_words['Jenis Kata'] = 'Kata Negatif'
top_words = pd.concat([all_pos_words, all_neg_words])

# Kanan — tambah label di bar top kata
sns.barplot(data=top_words, x='count', y='word',
            hue='Jenis Kata', ax=ax2, palette=['steelblue', 'salmon'])
ax2.set_title(f'Kata-kata yang Paling Dominan per Sentimen Berita', fontweight='bold')
ax2.set_xlabel('Frekuensi')
ax2.set_ylabel('')
ax2.spines[['top', 'right']].set_visible(False)
for container in ax2.containers:
    ax2.bar_label(container, fmt='%d', padding=3, fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('lm_lexicon_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
lexicon_data

# %%
avg_scores

# %%
avg_melted

# %% [markdown]
# ### Processing Text

# %%
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    import re
    import contractions
    from nltk.corpus import stopwords, wordnet
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag

    if not isinstance(text, str):
        return ""

    FINANCIAL_STOPWORDS_REMOVE = {
        'oyj', 'plc', 'ltd', 'inc', 'corp', 'omx', 'company',
        'finnish', 'finland', 'helsinki',
        'first', 'second', 'third', 'quarter',
        'euro', 'eur', 'million',
        'say', 'said', 'also', 'would', 'could', 'may', 'might',
        'one', 'two', 'three', 'new', 'year', 'week', 'day',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday'
    }

    FINANCIAL_KEEP = {
        'fed', 'ipo', 'gdp', 'cpi', 'sec', 'oil', 'gas', 'cut',
        'buy', 'sell', 'bid', 'ask', 'esg', 'etf', 'rba', 'imf',
        'up', 'down', 'not', 'low', 'high'
    }

    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(tag):
        if tag.startswith('J'): return wordnet.ADJ
        if tag.startswith('V'): return wordnet.VERB
        if tag.startswith('N'): return wordnet.NOUN
        if tag.startswith('R'): return wordnet.ADV
        return wordnet.NOUN

    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'\bmln\b', 'million', text)
    text = re.sub(r'\bbln\b', 'billion', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')) | FINANCIAL_STOPWORDS_REMOVE
    tokens = [t for t in tokens if t not in stop_words or t in FINANCIAL_KEEP]
    tokens = [t for t in tokens if len(t) > 2 or t in FINANCIAL_KEEP]
    tagged = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]

    return ' '.join(tokens)

pandarallel.initialize(nb_workers=4, progress_bar=True)

head_data['clean_head'] = head_data['column_2'].parallel_apply(preprocess_text)
head_data[['column_1', 'clean_head']].to_csv('after_prepro_head2.csv', index=False)

# %% [markdown]
# ### Word Frequency

# %%
data_hasil_prepro = 'after_prepro_head.csv'
prepro_data = pd.read_csv(data_hasil_prepro)
plot_hasil_prepro = plot_word_frequency(prepro_data, column='clean_head')

# %%
data_hasil_prepro1 = 'after_prepro_head1.csv'
prepro_data1 = pd.read_csv(data_hasil_prepro1)
plot_hasil_prepro1 = plot_word_frequency(prepro_data1, column='clean_head')

# %%
data_hasil_prepro2 = 'after_prepro_head2.csv'
prepro_data2 = pd.read_csv(data_hasil_prepro2)
# plot_hasil_prepro2 = plot_word_frequency(prepro_data2, column='clean_head')

# %% [markdown]
# ### N-Gram

# %% [markdown]
# #### Senua

# %%
def get_ngram_counter(series, n, extra_stop=None, batch_size=10000):
    counter = Counter()
    for i in range(0, len(series), batch_size):
        batch = series.iloc[i:i+batch_size].dropna()
        for text in batch:
            tokens = str(text).split()
            if extra_stop:
                tokens = [t for t in tokens if t not in extra_stop]
            counter.update(ngrams(tokens, n))
    return counter

def counter_to_freq(counter, top_k):
    top = counter.most_common(top_k)
    labels = [" ".join(gram) for gram, _ in top]
    counts = [count for _, count in top]
    return labels, counts

def counter_to_treemap(ax, counter, top_k, colormap="Blues"):
    top = counter.most_common(top_k)
    labels = [" ".join(gram) for gram, _ in top]
    sizes  = [count for _, count in top]

    norm   = plt.Normalize(vmin=min(sizes), vmax=max(sizes))
    cmap   = plt.get_cmap(colormap)
    colors = [cmap(norm(s)) for s in sizes]

    def text_color(rgba):
        r, g, b, _ = rgba
        luminance = 0.299*r + 0.587*g + 0.114*b
        return "white" if luminance < 0.5 else "black"

    squarify.plot(
        sizes=sizes,
        label=[f"{l}\n{s}" for l, s in zip(labels, sizes)],
        color=colors,
        ax=ax,
        text_kwargs={"fontsize": 7, "wrap": True},
        pad=True,
    )

    for text, rgba in zip(ax.texts, colors):
        text.set_color(text_color(rgba))

    ax.axis("off")

def counter_to_wordcloud(ax, counter, top_k, colormap="viridis", bg_color="white"):
    top = {" ".join(gram): count for gram, count in counter.most_common(top_k)}
    wc = WordCloud(
        width=800,
        height=400,
        background_color=bg_color,
        colormap=colormap,
        max_words=top_k,
    ).generate_from_frequencies(top)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")


# %%
# SETTING
n     = 2      # ganti 3 untuk trigram
top_k = 20

# Hitung counter
counter_head = get_ngram_counter(prepro_data2["clean_head"], n)
labels_head, counts_head = counter_to_freq(counter_head, top_k)

# Label otomatis
gram_label = "Bigram" if n == 2 else "Trigram" if n == 3 else f"{n}-gram"

# PLOT — 1 baris 3 kolom
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle(f"Analisis {gram_label} — clean_head", fontsize=16, fontweight="bold")

# Kolom 1 — Bar chart (seaborn)
sns.barplot(
    x=counts_head,
    y=labels_head,
    ax=ax1,
    palette="viridis"
)
ax1.set_title(f"Top {top_k} {gram_label} — Frekuensi", fontsize=12, fontweight="bold")
ax1.set_xlabel("Frekuensi")
ax1.set_ylabel("")
ax1.spines[["top", "right"]].set_visible(False)

# Kolom 2 — Treemap
counter_to_treemap(ax2, counter_head, top_k, colormap="YlOrRd")
ax2.set_title(f"Treemap {gram_label} — clean_head", fontsize=12, fontweight="bold")

# Kolom 3 — Wordcloud
counter_to_wordcloud(ax3, counter_head, top_k, colormap="viridis")
ax3.set_title(f"Wordcloud {gram_label} — clean_head", fontsize=12, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"ngram_{gram_label.lower()}_clean_head.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# SETTING
n     = 3
top_k = 20

# Hitung counter
counter_head = get_ngram_counter(prepro_data2["clean_head"], n)
labels_head, counts_head = counter_to_freq(counter_head, top_k)

# Label otomatis
gram_label = "Bigram" if n == 2 else "Trigram" if n == 3 else f"{n}-gram"

# PLOT — 1 baris 3 kolom
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle(f"Analisis {gram_label} — clean_head", fontsize=16, fontweight="bold")

# Kolom 1 — Bar chart (seaborn)
sns.barplot(
    x=counts_head,
    y=labels_head,
    ax=ax1,
    palette="viridis"
)
ax1.set_title(f"Top {top_k} {gram_label} — Frekuensi", fontsize=12, fontweight="bold")
ax1.set_xlabel("Frekuensi")
ax1.set_ylabel("")
ax1.spines[["top", "right"]].set_visible(False)

# Kolom 2 — Treemap
counter_to_treemap(ax2, counter_head, top_k, colormap="YlOrRd")
ax2.set_title(f"Treemap {gram_label} — clean_head", fontsize=12, fontweight="bold")

# Kolom 3 — Wordcloud
counter_to_wordcloud(ax3, counter_head, top_k, colormap="viridis")
ax3.set_title(f"Wordcloud {gram_label} — clean_head", fontsize=12, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"ngram_{gram_label.lower()}_clean_head.png", dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# #### Positive, Negative, dan Neutral

# %%
# SETTING
n     = 3      # ganti 3 untuk trigram
top_k = 20

# Hitung counter
sentimen_label = 'positive'
counter_head = get_ngram_counter(prepro_data2[prepro_data2['column_1'] == sentimen_label]['clean_head'], n)
labels_head, counts_head = counter_to_freq(counter_head, top_k)

# Label otomatis
gram_label = "Bigram" if n == 2 else "Trigram" if n == 3 else f"{n}-gram"

# PLOT — 1 baris 3 kolom
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(22, 8))
fig.suptitle(f"Top Frasa yang Paling Sering Muncul pada Sentimen {sentimen_label}", fontsize=16, fontweight="bold")

# Kolom 1 — Bar chart (seaborn)
sns.barplot(x=counts_head, y=labels_head, ax=ax1, palette="viridis")
for container in ax1.containers:
    ax1.bar_label(container, padding=3, fontsize=14, fontweight='bold')
ax1.set_title(f"Top {top_k} {gram_label} — Frekuensi", fontsize=12, fontweight="bold")
ax1.set_xlabel("Frekuensi")
ax1.set_ylabel("")
ax1.spines[["top", "right"]].set_visible(False)

# # Kolom 2 — Treemap
# counter_to_treemap(ax2, counter_head, top_k, colormap="YlOrRd")
# ax2.set_title(f"Treemap {gram_label} — clean_head", fontsize=12, fontweight="bold")

# Kolom 3 — Wordcloud
counter_to_wordcloud(ax3, counter_head, top_k, colormap="viridis")
ax3.set_title(f"Wordcloud {gram_label} — clean_head", fontsize=12, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
# plt.savefig(f"ngram_{gram_label.lower()}_clean_head.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# SETTING
n     = 3      # ganti 3 untuk trigram
top_k = 20

# Hitung counter
sentimen_label = 'negative'
counter_head = get_ngram_counter(prepro_data2[prepro_data2['column_1'] == sentimen_label]['clean_head'], n)
labels_head, counts_head = counter_to_freq(counter_head, top_k)

# Label otomatis
gram_label = "Bigram" if n == 2 else "Trigram" if n == 3 else f"{n}-gram"

# PLOT — 1 baris 3 kolom
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(22, 8))
fig.suptitle(f"Top Frasa yang Paling Sering Muncul pada Sentimen {sentimen_label}", fontsize=16, fontweight="bold")

# Kolom 1 — Bar chart (seaborn)
sns.barplot(x=counts_head, y=labels_head, ax=ax1, palette="viridis")
for container in ax1.containers:
    ax1.bar_label(container, padding=3, fontsize=14, fontweight='bold')
ax1.set_title(f"Top {top_k} {gram_label} — Frekuensi", fontsize=12, fontweight="bold")
ax1.set_xlabel("Frekuensi")
ax1.set_ylabel("")
ax1.spines[["top", "right"]].set_visible(False)

# # Kolom 2 — Treemap
# counter_to_treemap(ax2, counter_head, top_k, colormap="YlOrRd")
# ax2.set_title(f"Treemap {gram_label} — clean_head", fontsize=12, fontweight="bold")

# Kolom 3 — Wordcloud
counter_to_wordcloud(ax3, counter_head, top_k, colormap="viridis")
ax3.set_title(f"Wordcloud {gram_label} — clean_head", fontsize=12, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
# plt.savefig(f"ngram_{gram_label.lower()}_clean_head.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# SETTING
n     = 3      # ganti 3 untuk trigram
top_k = 20

# Hitung counter
sentimen_label = 'neutral'
counter_head = get_ngram_counter(prepro_data2[prepro_data2['column_1'] == sentimen_label]['clean_head'], n)
labels_head, counts_head = counter_to_freq(counter_head, top_k)

# Label otomatis
gram_label = "Bigram" if n == 2 else "Trigram" if n == 3 else f"{n}-gram"

# PLOT — 1 baris 3 kolom
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(22, 8))
fig.suptitle(f"Top Frasa yang Paling Sering Muncul pada Sentimen {sentimen_label}", fontsize=16, fontweight="bold")

# Kolom 1 — Bar chart (seaborn)
sns.barplot(x=counts_head, y=labels_head, ax=ax1, palette="viridis")
for container in ax1.containers:
    ax1.bar_label(container, padding=3, fontsize=14, fontweight='bold')
ax1.set_title(f"Top {top_k} {gram_label} — Frekuensi", fontsize=12, fontweight="bold")
ax1.set_xlabel("Frekuensi")
ax1.set_ylabel("")
ax1.spines[["top", "right"]].set_visible(False)

# # Kolom 2 — Treemap
# counter_to_treemap(ax2, counter_head, top_k, colormap="YlOrRd")
# ax2.set_title(f"Treemap {gram_label} — clean_head", fontsize=12, fontweight="bold")

# Kolom 3 — Wordcloud
counter_to_wordcloud(ax3, counter_head, top_k, colormap="viridis")
ax3.set_title(f"Wordcloud {gram_label} — clean_head", fontsize=12, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
# plt.savefig(f"ngram_{gram_label.lower()}_clean_head.png", dpi=300, bbox_inches='tight')
plt.show()
