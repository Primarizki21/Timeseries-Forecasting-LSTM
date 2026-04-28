# bank-stock-forecast

Autonomous ML research agent untuk forecasting harga saham bank Indonesia (BBCA, BBRI, BMRI) dengan LSTM.

## Dataset

- **Task**: Forecast harga closing (`close`) per saham — univariate/multivariate time series regression.
- **Saham**: BBCA (BCA), BBRI (BRI), BMRI (Mandiri) — masing-masing punya model LSTM sendiri.
- **Periode**: **2 tahun terakhir (WAJIB ambil via yfinance, tidak menggunakan CSV lokal)**
- **Split**: Train 70% / Val 15% / Test 15% — **chronological, tidak di-shuffle**.
- **Data Source**: `yfinance` (ticker: BBCA.JK, BBRI.JK, BMRI.JK)

### Data Loading Strategy (MANDATORY)

1. Cek folder `data_saham/`, bila tidak ada, buat foldernya.
2. Untuk setiap ticker:
   - Jika file CSV tersedia → load dari CSV
   - Jika TIDAK ada → fetch dari `yfinance`, lalu simpan ke CSV
3. File harus disimpan sebagai:
   - `data_saham/BBCA.csv`
   - `data_saham/BBRI.csv`
   - `data_saham/BMRI.csv`

### Optional Refresh Policy

- Jika file lebih lama dari 1 hari → fetch ulang dari yfinance

### Kolom target & fitur

| Kolom | Keterangan |
|---|---|
| `close` | Target utama (harga penutupan) |
| `open`, `high`, `low`, `volume` | Fitur tambahan opsional |
| MA_20 | Moving average 20 hari |
| RSI_14 | Relative Strength Index 14 hari |
| log_return | `log(close_t / close_t-1)` |
| lag_1, lag_2 | Nilai close 1 dan 2 hari sebelumnya |

> **Catatan**: Gunakan **Polars** untuk komputasi (rolling, lag, log return). Gunakan **pandas** hanya sebagai jembatan ke matplotlib/seaborn untuk plotting. Feature engineering dilakukan sebelum scaling.

## Primary metric: **RMSE (lower is better)**

Target:
- **Minimum acceptable**: RMSE < 5
- **Good**: RMSE < 4
- **Excellent**: RMSE < 3

Goal utama agent adalah mencapai RMSE serendah mungkin, dengan prioritas mencapai threshold "Excellent".

---

## Setup

1. **Agree on a run tag**: propose tag berdasarkan tanggal hari ini (e.g. `apr28`).
2. **Create branch**: `git checkout -b autoresearch/<tag>`
3. **Read these files**: `prepare.py`, `experiment.py`, `program.md`
4. **Verify data**: pastikan file CSV atau akses `yfinance` tersedia untuk BBCA, BBRI, BMRI.
5. **Reconstruct experiment memory jika perlu**: Cek apakah `experiment_memory.md` ada dan tidak kosong.
   - **Jika ada dan berisi**: baca dulu — berisi history semua eksperimen sebelumnya. Jangan ulangi yang sudah ada.
   - **Jika kosong atau tidak ada**: rekonstruksi dari git history sebelum mulai:
     1. Jalankan `git log --oneline`
     2. Untuk tiap commit, jalankan `git show <commit> -- experiment.py`
     3. Baca `evaluation_output/<commit>/experiment_card.txt` jika tersedia
     4. Rekonstruksi dan append ke `experiment_memory.md` sesuai format di bawah
     5. Setelah semua rekonstruksi selesai, lanjut normal
   - **Jika belum ada commit sama sekali**: mulai dengan EDA dulu, lalu baseline LSTM.
6. **Initialize results.tsv**: buat dengan header saja jika belum ada.
7. **Confirm dan mulai**.

---

## Fase 1 — EDA (wajib dilakukan sebelum eksperimen model)

EDA harus dijalankan sekali untuk **masing-masing saham** (BBCA, BBRI, BMRI). Semua plot disimpan ke `evaluation_output/eda/`.

### Tools
- **Polars**: komputasi rolling MA, log return, lag
- **pandas**: konversi ke DataFrame untuk plotting
- **seaborn / matplotlib**: visualisasi (utamakan seaborn)

### Plot yang harus dibuat per saham

#### 1. Time Series Decomposition
- Dekomposisi `close` menjadi komponen: **trend, seasonal, residual**
- Gunakan `statsmodels.tsa.seasonal.seasonal_decompose` (model additive atau multiplicative)
- Simpan sebagai `eda/decomposition_<TICKER>.png`

#### 2. Harga Closing + Moving Average
Buat dua jenis plot:

**a. All-in-one (overlap)**
- Satu plot dengan harga `close` asli + MA_10, MA_20, MA_50, MA_100 di-overlay
- Warna berbeda per MA, legend jelas
- Simpan: `eda/ma_overlay_<TICKER>.png`

**b. Subplots (per-MA panel)**
- 4 panel: setiap panel berisi harga asli + satu jenis MA saja
- Panel: [MA_10], [MA_20], [MA_50], [MA_100]
- Simpan: `eda/ma_subplots_<TICKER>.png`

Komputasi MA menggunakan Polars:
```python
df = df.with_columns([
    pl.col("close").rolling_mean(10).alias("MA_10"),
    pl.col("close").rolling_mean(20).alias("MA_20"),
    pl.col("close").rolling_mean(50).alias("MA_50"),
    pl.col("close").rolling_mean(100).alias("MA_100"),
])
```

#### 3. Uji Stasioneritas (ADF Test)
- Jalankan ADF test pada `close` untuk tiap saham
- Jika **p-value > 0.05** (non-stasioner) → lakukan differencing (gunakan `log_return` atau first difference `close.diff()`) sebelum masuk LSTM
- Catat hasil di `eda/adf_results.txt`:
  ```
  BBCA: ADF stat=-1.23, p=0.6512 → NON-STATIONARY → gunakan log_return
  BBRI: ADF stat=-3.45, p=0.0091 → STATIONARY
  BMRI: ADF stat=-2.11, p=0.2340 → NON-STATIONARY → gunakan log_return
  ```
- Simpan: `eda/adf_results.txt`

#### 4. Plot ACF dan PACF
- Untuk tiap saham: buat **2 plot** terpisah — ACF dan PACF
- Gunakan `statsmodels.graphics.tsaplots.plot_acf` dan `plot_pacf`
- Lags: tampilkan hingga 40 lag
- **Hasil ini digunakan untuk menentukan `look_back` (window size) model LSTM**
  - Jika ACF cut-off di lag 5 → coba `look_back=5`
  - Jika ada seasonal pattern → jadikan kandidat window size
- Simpan: `eda/acf_<TICKER>.png`, `eda/pacf_<TICKER>.png`

### Catatan EDA → Model
Setelah EDA selesai, tentukan dan catat:
- `look_back` kandidat per saham berdasarkan ACF/PACF
- Apakah perlu differencing berdasarkan ADF
- Fitur mana yang akan digunakan di baseline

---

## Fase 2 — Preprocessing (`prepare.py`)

`prepare.py` adalah **fixed harness** — jangan dimodifikasi kecuali bagian feature engineering.

### Pipeline preprocessing (urutan wajib)
1. Load data CSV / yfinance untuk periode 2 tahun
2. Sort by date (chronological)
3. Komputasi fitur dengan **Polars**:
   - `log_return = log(close / close.shift(1))`
   - `MA_20 = close.rolling_mean(20)`
   - `RSI_14` (lihat formula di bawah)
   - `lag_1 = close.shift(1)`, `lag_2 = close.shift(2)`
4. Drop NaN rows (akibat rolling/lag)
5. Split chronological: train 70% / val 15% / test 15%
6. **Scale**: gunakan `MinMaxScaler` atau `StandardScaler` pada fitur numerik — fit **hanya pada train set**, transform val dan test
7. Buat sequences LSTM: `(X, y)` dengan shape `(n_samples, look_back, n_features)`

### Formula RSI (Polars)
```python
delta = pl.col("close").diff()
gain = delta.clip(lower_bound=0).rolling_mean(14)
loss = (-delta).clip(lower_bound=0).rolling_mean(14)
rs = gain / loss
rsi = 1 - (1 / (1 + rs))
df = df.with_columns(rsi.alias("RSI_14"))
```

### Split chronological
```python
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
train = df[:train_end]
val   = df[train_end:val_end]
test  = df[val_end:]
```

### Sequence builder
```python
def make_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])  # (look_back, n_features)
        y.append(data[i, 0])           # target = close (kolom 0)
    return np.array(X), np.array(y)
```

**DO NOT MODIFY**: fungsi `evaluate()` dan split ratio.

---

## Fase 3 — Eksperimen Model (`experiment.py`)

### Baseline LSTM
```
Input shape : (look_back, n_features)
Architecture: LSTM(64) → Dropout(0.2) → Dense(1)
Optimizer   : Adam lr=0.001
Loss        : MSE
Epochs      : 50 (dengan EarlyStopping patience=10)
Batch size  : 32
```

### Feature engineering baseline
- RSI_14, MA_20, log_return, lag_1

### Eksperimen yang dianjurkan (urutan prioritas)
- Variasi `look_back`: 5, 10, 20, 30 (berdasarkan hasil ACF/PACF)
- Tambah/kurangi fitur: lag_2, volume, MA_50
- Arsitektur: LSTM 2 layer, BiLSTM, LSTM+Attention
- Hyperparameter: hidden units, dropout rate, learning rate
- `USE_LOG_RETURN = True/False`: forecast log_return lalu inverse ke harga
- Batch size: 16, 32, 64
- Optimizer: Adam vs RMSprop

### Satu model per saham
Tiap saham (BBCA, BBRI, BMRI) dilatih **secara independen** dengan model terpisah. Jangan gabungkan data antar saham.

---

## Evaluation Outputs

Untuk setiap eksperimen, simpan ke `evaluation_output/<commit_hash>/`.

Artifacts wajib:
1. **Prediction plot** (`prediction_<TICKER>.png`) — plot time series: actual (groundtruth) vs predicted pada test set, untuk setiap saham
2. **Training history** (`training_history_<TICKER>.png`) — kurva train loss vs val loss per epoch
3. **Regression report** (`regression_report.csv` dan `.xlsx`) — RMSE dan MAE per saham
4. **Experiment card** (`experiment_card.txt`)

Jika `status = keep`, tambahkan juga:
5. **Model file** per saham: `model_<TICKER>_<slug>.keras` atau `model_<TICKER>_<slug>.pth`
6. **Submission CSV** (opsional untuk kompetisi): prediksi test set per saham

### Format prediction plot

```python
plt.figure(figsize=(14, 5))
plt.plot(dates_test, y_true, label="Actual (Groundtruth)", color="steelblue")
plt.plot(dates_test, y_pred, label="Predicted", color="tomato", linestyle="--")
plt.title(f"LSTM Forecast — {ticker} | RMSE={rmse:.4f}")
plt.legend()
plt.tight_layout()
plt.savefig(f"evaluation_output/{commit}/prediction_{ticker}.png")
```

> Plot ini wajib ada agar bisa melihat secara visual seberapa jauh prediksi dari harga aslinya, bukan hanya mengandalkan angka RMSE.

### Experiment Card format (`experiment_card.txt`)

```
=== EXPERIMENT CARD ===
commit:         a1b2c3d
date:           2025-04-28 14:00
branch:         autoresearch/apr28

--- Model ---
type:           LSTM (2 layer)
architecture:   LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.2) → Dense(1)
optimizer:      Adam lr=0.001
epochs_run:     47 (early stopped at 47/100)

--- Preprocessing ---
scaler:         MinMaxScaler (fit on train only)
look_back:      20
use_log_return: False
features:       close, RSI_14, MA_20, log_return, lag_1, lag_2

--- Feature Engineering ---
- RSI_14: Relative Strength Index 14 hari
- MA_20: Moving Average 20 hari
- log_return: log(close_t / close_t-1)
- lag_1, lag_2: close geser 1 dan 2 hari

--- Results ---
BBCA: rmse=3.21, mae=2.54
BBRI: rmse=4.87, mae=3.91
BMRI: rmse=4.12, mae=3.20
rmse_avg: 4.07
status:   keep  |  discard  |  crash
model_files:
  BBCA: model_BBCA_lstm2l_fe.keras
  BBRI: model_BBRI_lstm2l_fe.keras
  BMRI: model_BMRI_lstm2l_fe.keras

--- Why tried ---
Hipotesis: 2-layer LSTM lebih mampu menangkap pola non-linear jangka menengah.

--- What worked / didn't ---
(isi setelah lihat hasil)
```

---

## Output format (stdout)

```
---
ticker:          BBCA
rmse:            3.210000
mae:             2.540000
ticker:          BBRI
rmse:            4.870000
mae:             3.910000
ticker:          BMRI
rmse:            4.120000
mae:             3.200000
rmse_avg:        4.067000
training_seconds:87.3
model:           LSTM 2-layer look_back=20 fe=RSI+MA+logret+lag
```

---

## Logging results

Log ke `results.tsv` (tab-separated):

```
commit	rmse_bbca	rmse_bbri	rmse_bmri	rmse_avg	status	description
```

Contoh:
```
commit	rmse_bbca	rmse_bbri	rmse_bmri	rmse_avg	status	description
a1b2c3d	6.51	7.20	6.88	6.86	keep	baseline LSTM 1-layer look_back=10
b2c3d4e	3.21	4.87	4.12	4.07	keep	2-layer LSTM look_back=20 RSI+MA+logret+lag
c3d4e5f	9.99	9.99	9.99	9.99	crash	BiLSTM — import error
```

---

## Experiment Memory (`experiment_memory.md`)

Append satu blok per eksperimen. File ini **tidak di-commit ke git**.

### Format

```markdown
### [commit: a1b2c3d] LSTM baseline look_back=10 — KEEP (rmse_avg: 6.86)
- Model: LSTM(64) → Dropout(0.2) → Dense(1), epoch=50
- look_back: 10
- Features: close, RSI_14, MA_20, log_return, lag_1
- Scaler: MinMaxScaler
- Result: BBCA=6.51, BBRI=7.20, BMRI=6.88, avg=6.86
- Model files: model_BBCA_lstm1l.keras, model_BBRI_lstm1l.keras, model_BMRI_lstm1l.keras
- Notes: Baseline established. BBRI paling susah diprediksi (volatilitas tinggi).

### [commit: b2c3d4e] LSTM 2-layer look_back=20 — KEEP (rmse_avg: 4.07)
- Model: LSTM(128)→Dropout(0.3)→LSTM(64)→Dropout(0.2)→Dense(1)
- look_back: 20
- Features: close, RSI_14, MA_20, log_return, lag_1, lag_2
- Scaler: MinMaxScaler
- Result: BBCA=3.21, BBRI=4.87, BMRI=4.12, avg=4.07
- Model files: model_BBCA_lstm2l_fe.keras, model_BBRI_lstm2l_fe.keras, model_BMRI_lstm2l_fe.keras
- Notes: Signifikan lebih baik. look_back=20 sesuai ACF. lag_2 sedikit membantu BMRI.
```

**Rules**:
- Jangan hapus entry, hanya append
- Selalu tulis hipotesis DAN hasilnya
- Jika crash, log tetap dibuat dengan status "CRASH" dan jenis error-nya
- Jangan di-commit ke git (add ke `.gitignore`)
- Baca file ini dulu sebelum propose eksperimen baru

---

## The Experiment Loop

LOOP FOREVER:

1. Cek git state (branch/commit saat ini)
2. Baca `experiment_memory.md` — pilih ide yang **belum dicoba**
3. Modifikasi `experiment.py` dengan ide baru
4. Jika perlu library baru: `uv add <package>`
5. `git commit`
6. Jalankan:
   ```
   uv run experiment.py > run.log 2>&1
   ```
7. Baca hasil: `grep "^rmse_avg:" run.log`
8. Jika grep kosong → crash. Jalankan `tail -n 30 run.log` untuk stack trace.
9. Simpan evaluation artifacts ke `evaluation_output/<commit>/`
10. Jika `rmse_avg` turun (lebih baik) → **keep**:
    - Simpan model per saham ke `evaluation_output/<commit>/model_<TICKER>_<slug>.keras`
    - Keep commit, advance branch
11. Jika tidak ada improvement → `git reset --hard HEAD~1` (tapi tetap tulis memory entry!)
12. Log ke `results.tsv`
13. Append ke `experiment_memory.md`

---

## Struktur Folder

```
bank-stock-forecast/
├── program.md              ← dokumen ini
├── prepare.py              ← fixed harness (jangan dimodifikasi)
├── experiment.py           ← file yang dimodifikasi agent
├── data_saham/
│   ├── BBCA.csv
│   ├── BBRI.csv
│   └── BMRI.csv
├── evaluation_output/
│   ├── eda/                ← semua plot EDA
│   │   ├── decomposition_BBCA.png
│   │   ├── ma_overlay_BBCA.png
│   │   ├── ma_subplots_BBCA.png
│   │   ├── acf_BBCA.png
│   │   ├── pacf_BBCA.png
│   │   ├── adf_results.txt
│   │   └── ... (sama untuk BBRI, BMRI)
│   └── <commit_hash>/      ← artifacts per eksperimen
│       ├── prediction_BBCA.png
│       ├── prediction_BBRI.png
│       ├── prediction_BMRI.png
│       ├── training_history_BBCA.png
│       ├── regression_report.csv
│       ├── regression_report.xlsx
│       ├── experiment_card.txt
│       └── model_<TICKER>_<slug>.keras (jika keep)
├── experiment_memory.md    ← jangan di-commit!
├── results.tsv
└── run.log
```

---

## Hardware & Time Budget

- Gunakan GPU jika tersedia (`device='cuda'` untuk PyTorch, atau TensorFlow dengan GPU support)
- Tiap eksperimen (3 saham) harus selesai dalam **< 10 menit**
- Jika lebih dari 10 menit → kill, treat as failure, catat di memory
- Time budget per saham: ~3 menit

---

## Ideas to Explore (urutan prioritas)

- `look_back` variations: 5, 10, 20, 30 (gunakan hasil ACF/PACF sebagai anchor)
- Feature ablation: mana yang paling berpengaruh (RSI? MA? lag?)
- 2-layer LSTM, 3-layer LSTM
- Bidirectional LSTM
- GRU (lebih ringan dari LSTM)
- LSTM + Attention mechanism
- `USE_LOG_RETURN = True`: prediksi log return lalu inverse ke harga
- Dropout tuning: 0.1, 0.2, 0.3, 0.5
- Learning rate scheduler (ReduceLROnPlateau)
- Batch size: 16, 32, 64
- Feature: tambah `volume` jika tersedia
- Target encoding `day_of_week` sebagai fitur siklikal (sin/cos)

---

## Stop Conditions (Optional Override)

Meskipun default behavior adalah autonomous loop tanpa henti, agent BOLEH berhenti jika salah satu kondisi berikut terpenuhi:

1. **RMSE sudah sangat baik**:
   - rmse_avg < 3 (Excellent threshold)

2. **Tidak ada improvement signifikan**:
   - Tidak ada penurunan RMSE dalam 10 eksperimen berturut-turut

3. **Time / resource constraint**:
   - Sudah mencapai 50 eksperimen

Jika kondisi ini terpenuhi:
- Simpan hasil terbaik
- Tulis summary di `experiment_memory.md`
- Stop loop secara graceful

---

**Default behavior**: loop terus secara autonomous.

**Exception**: boleh berhenti jika memenuhi Stop Conditions di atas.