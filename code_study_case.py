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
#     display_name: percobaan (3.12.13)
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
import seaborn as sns
import numpy as np
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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
prepro1_housing = pd.get_dummies(prepro1_housing, columns=['furnishingstatus'], drop_first=True)
display(prepro1_housing)

# %% [markdown]
# ### EDA

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
plt.figure(figsize=(7, 5))
plt.scatter(prepro1_housing['area'], prepro1_housing['price'], alpha=0.4, color='steelblue')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.tight_layout()
plt.savefig('03_area_vs_price.png', dpi=100)
plt.show()

# %% [markdown]
# ## Time-Series Forecasting

# %% [markdown]
# ### Load Dataset

# %%
saham_tesla = yf.Ticker('TSLA')
print(saham_tesla.info['longBusinessSummary'])

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
# ## Sentiment Analysis

# %% [markdown]
#
