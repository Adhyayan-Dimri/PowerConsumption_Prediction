#  Power Consumption Prediction — XGBoost + Optuna

A context-aware 15-minute interval power load forecasting system built in Python.  
Trains separate XGBoost models per season, time-of-day period, and day type (weekday/weekend) for maximum accuracy.

---

##  Project Structure

```
Power Prediction/
│
├── power_xgboost.py        ← main script (the only file you run)
├── Dataset.xlsx            ← your input data (must be in this same folder)
├── .gitignore              ← excludes Dataset.xlsx from version control
└── Power_Prediction_YYYY-MM-DD_to_YYYY-MM-DD.xlsx   ← output (auto-generated here)
```

> **Both `Dataset.xlsx` and the script must be in the same folder.**  
> The output Excel file is also saved in the same folder automatically.


##  Input File — `Dataset.xlsx`

| Column | Format | Description |
|---|---|---|
| `DateTime` | `YYYY-MM-DD HH:MM:SS` | Timestamp of each reading |
| `Load_MW` | Decimal number | Power consumption in Megawatts |

- Data must be at **15-minute intervals** (96 readings per day)
- Minimum recommended data: **1 full year**
- No gaps or missing timestamps (the script drops rows with missing `Load_MW`)

**Example:**
| DateTime | Load_MW |
|---|---|
| 2025-04-01 00:00:00 | 412.5 |
| 2025-04-01 00:15:00 | 408.2 |
| 2025-04-01 00:30:00 | 401.7 |

---

##  Output File

**Filename:** `Power_Prediction_<start_date>_to_<end_date>.xlsx`  
**Saved to:** Same folder as the script

**Example:** `Power_Prediction_2026-04-01_to_2026-04-03.xlsx`

### Sheets inside the output file:

| Sheet | Contents |
|---|---|
| `All Predictions` | All 15-min slots across all predicted days in one table + daily summary |
| `01 Apr`, `02 Apr`, … | One sheet per predicted day with load table, summary strip, and line chart |
| `Model Performance` | Validation metrics (MAE, RMSE, MAPE) + best hyperparameters per context |
| `Tuning History` | Every Optuna trial result — best trial highlighted in green |

---

## ⚙️ Configuration — What You Can Change

All settings are at the **top of the script** (lines 8–22):

```python
PREDICT_DAYS = 3            # Number of days to predict
MONTH_WINDOW = 1            # ± months around target month for training data
START_DATE   = "2026-04-01" # Start date for prediction (see details below)
N_TRIALS     = 300          # Optuna tuning trials per model (higher = more accurate, slower)
```

### `PREDICT_DAYS`
How many consecutive days to predict.
```python
PREDICT_DAYS = 1   # predict only tomorrow
PREDICT_DAYS = 7   # predict next 7 days
```

### `START_DATE`
The first day you want predictions for.
```python
START_DATE = "2026-06-04"   # predict starting June 4, 2026
START_DATE = None           # auto: predict starting the day after last date in your data
```
> Format must be `"YYYY-MM-DD"` with quotes.

### `MONTH_WINDOW`
Controls how many surrounding months are included in training context.  
`MONTH_WINDOW = 1` means if predicting June, trains on May + June + July data.
```python
MONTH_WINDOW = 0   # strictly same month only
MONTH_WINDOW = 1   # ±1 month (recommended)
MONTH_WINDOW = 2   # ±2 months (more data, less seasonal precision)
```

### `N_TRIALS`
Number of Optuna hyperparameter search trials per context model.

| Value | Speed | Accuracy |
|---|---|---|
| 50 | ~2–5 min | Good |
| 150 | ~8–15 min | Better |
| 300 | ~20–40 min | Best |

---

##  How the Model Works

### Step 1 — Context Filtering
Instead of training one model on all data, the script trains a **separate XGBoost model for each unique combination** of:

- **Season/Month context** — e.g. predicting April trains only on March+April+May historical data
- **Time of day** — Night (00–05h), Morning (06–11h), Afternoon (12–17h), Evening (18–23h)
- **Day type** — Weekday (Mon–Fri) vs Weekend (Sat–Sun)

So a "April Afternoon Weekday" slot is predicted by a model that has only ever seen April afternoon weekday data — the most relevant patterns for that specific context.

### Step 2 — Feature Engineering
The model uses 27 features built from the DateTime and historical load:

| Feature Group | Features |
|---|---|
| Calendar | hour, minute, day-of-week, month, quarter, day-of-year, week-of-year |
| Cyclic encoding | sin/cos transforms of hour, slot, day-of-week, month |
| Lag features | Load from 1, 2, 3, 7, 14 days ago at same time slot |
| Rolling stats | 1-day and 7-day rolling mean and std |
| Slot statistics | Historical average and std at each 15-min slot of the day |

### Step 3 — Optuna Hyperparameter Tuning
For each context model, Optuna runs `N_TRIALS` combinations of XGBoost parameters using **TimeSeriesSplit cross-validation** (3 folds, temporal order preserved). It picks the combination with the lowest RMSE.

Parameters tuned: `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`

### Step 4 — Iterative Prediction
For multi-day prediction, the model predicts **one slot at a time** and feeds each prediction back as lag history for the next slot. This means Day 2 predictions use Day 1 predicted values as lag features.

### Step 5 — Fallback Logic
If a context has fewer than 200 rows of training data:
1. First fallback: relax the day-type filter (mix weekday + weekend)
2. Second fallback: use all data for that time period (ignore month filter)

This ensures the model never crashes on thin data.

---

##  Requirements

```bash
pip install pandas numpy xgboost optuna scikit-learn openpyxl
```

| Library | Version | Purpose |
|---|---|---|
| pandas | ≥ 2.0 | Data loading and manipulation |
| numpy | ≥ 1.24 | Numerical operations |
| xgboost | ≥ 2.0 | Gradient boosting model |
| optuna | ≥ 3.0 | Hyperparameter tuning |
| scikit-learn | ≥ 1.3 | TimeSeriesSplit cross-validation |
| openpyxl | ≥ 3.1.5 | Excel output generation |

---

##  How to Run

1. Place `Dataset.xlsx` and `power_xgboost.py` in the same folder
2. Open terminal / PowerShell in that folder
3. Run:
```bash
python power_xgboost.py
```

### Expected Console Output
```
============================================================
 Power Prediction — XGBoost + Optuna
============================================================

[1/6] Loading Dataset.xlsx ...
  Records : 35,040
  Range   : 2025-04-01 → 2026-03-31
  Load    : 312.4 – 891.2 MW

[2/6] Engineering features ...

[3/6] Tuning 8 context model(s)  (300 Optuna trials each) ...
      Predicting: 2026-04-01 → 2026-04-03

  [1/8]  Month=Apr  Period=Afternoon   Type=weekday    1,234 rows  → tuning ...  ✓  CV-RMSE=18.4 MW  MAE=14.2 MW  MAPE=2.8%
  ...

[4/6] Generating 3-day predictions ...
  Day 1 (2026-04-01) [Weekday]  —  avg=612.3 MW  peak=847.1 MW  energy=6,123.0 MWh
  Day 2 (2026-04-02) [Weekday]  —  avg=608.7 MW  peak=831.4 MW  energy=6,087.0 MWh
  Day 3 (2026-04-03) [Weekday]  —  avg=601.2 MW  peak=819.6 MW  energy=6,012.0 MWh

[5/6] Writing → Power_Prediction_2026-04-01_to_2026-04-03.xlsx ...

[6/6] Done!  Saved → Power_Prediction_2026-04-01_to_2026-04-03.xlsx
```

---

##  Understanding the Accuracy Metrics

The script reports three metrics evaluated on a held-out validation set (last 7 days of each context):

| Metric | Full Name | What it means | Good range |
|---|---|---|---|
| **MAE** | Mean Absolute Error | Average MW error per slot | < 20 MW |
| **RMSE** | Root Mean Squared Error | Penalises large errors more | < 30 MW |
| **MAPE** | Mean Absolute Percentage Error | % error relative to actual load | < 5% |

> **MAPE is the most intuitive** — a MAPE of 3% means predictions are off by 3% on average.  
> 95% accuracy = 5% MAPE. 98% accuracy = 2% MAPE.

---

## 🔧 Common Issues & Fixes

### `ModuleNotFoundError: No module named 'pandas'`
```bash
python -m pip install pandas numpy xgboost optuna scikit-learn openpyxl
```

### `ImportError: Pandas requires version '3.1.5' or newer of 'openpyxl'`
```bash
python -m pip install --upgrade openpyxl
```

### `KeyError: 'slot_mean' not in index`
Your Dataset.xlsx is likely missing data or has a column name mismatch. Check that columns are named exactly `DateTime` and `Load_MW` (case-sensitive).

### Prediction date is wrong / predicting old dates
Set `START_DATE` explicitly in the config:
```python
START_DATE = "2026-06-04"
```

### Script runs but accuracy is low (< 90%)
- Increase `N_TRIALS` to 300+
- Check that your dataset has at least 6 months of data
- Ensure no large gaps in the DateTime column

---

##  Changing Prediction Dates — Quick Reference

| Goal | Change |
|---|---|
| Predict tomorrow (auto) | `START_DATE = None` |
| Predict specific date | `START_DATE = "2026-06-04"` |
| Predict 1 day | `PREDICT_DAYS = 1` |
| Predict next week | `PREDICT_DAYS = 7` |
| Predict a month | `PREDICT_DAYS = 30` |

---

##  Notes

- The script is **entirely offline** — no internet connection required
- Output Excel file is **overwritten** if you run with the same date range again
- For predictions far into the future (e.g. 1+ year ahead), lag features will be filled with historical medians since actual recent data won't exist — accuracy will be lower than near-term predictions
- Weekend detection is automatic based on Python's calendar (Saturday = 5, Sunday = 6) — no configuration needed
- The `Tuning History` sheet in the output Excel lets you inspect every Optuna trial; the best trial row is highlighted in green

---

*Built for 15-minute interval power load forecasting in Indian grid context.*
