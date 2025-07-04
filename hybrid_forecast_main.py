
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import root_mean_squared_error

# Configuration
DATA_FILE = 'air_quality_data.xlsx'
REQUIRED_COLUMNS = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
    'PT08.S5(O3)', 'T', 'RH', 'AH'
]
VOLATILE_COLUMNS = ['PT08.S3(NOx)', 'RH', 'PT08.S5(O3)', 'PT08.S4(NO2)']
WINDOW_SIZE = 5

# Load and clean
raw_df = pd.read_excel(DATA_FILE, sheet_name=0)
df = raw_df.loc[:, ~raw_df.columns.str.contains('^Unnamed')]
df.replace(-200, np.nan, inplace=True)

df['Datetime'] = pd.to_datetime(
    df['Date'].astype(str) + ' ' + df['Time'].astype(str),
    errors='coerce', dayfirst=True
)
df.dropna(subset=['Datetime'], inplace=True)
df.set_index('Datetime', inplace=True)
df.drop(['Date', 'Time'], axis=1, inplace=True)

df_clean = df.ffill().bfill()

# Add time features
df_clean['hour'] = df_clean.index.hour
df_clean['day'] = df_clean.index.day
df_clean['month'] = df_clean.index.month
df_clean['day_of_week'] = df_clean.index.dayofweek

# RMSE on last 10% using hybrid strategy
n = len(df_clean)
test_size = int(n * 0.1)
train = df_clean.iloc[:-test_size]
test = df_clean.iloc[-test_size:]

results = {}

print('\nHybrid Forecast RMSE on last 10%:')
for col in REQUIRED_COLUMNS:
    y_true = test[col].values

    # Hybrid prediction strategy
    if col in VOLATILE_COLUMNS:
        y_pred_value = train[col].iloc[-1]
    else:
        y_pred_value = train[col].iloc[-WINDOW_SIZE:].mean()

    y_pred = [y_pred_value] * len(y_true)

    if np.all(y_true == y_true[0]):
        print(f"{col}: WARNING - constant or missing test data")
        rmse = 0.0
    else:
        rmse = root_mean_squared_error(y_true, y_pred)

    results[col] = rmse
    print(f"{col}: {rmse}")

# Forecast next 48 hours
forecast_48 = {}
for col in REQUIRED_COLUMNS:
    if col in VOLATILE_COLUMNS:
        forecast_value = df_clean[col].iloc[-1]
    else:
        forecast_value = df_clean[col].iloc[-WINDOW_SIZE:].mean()
    forecast_48[col] = [forecast_value] * 48

forecast_df = pd.DataFrame(forecast_48)
forecast_df.index = pd.date_range(
    start=df_clean.index[-1] + pd.Timedelta(hours=1),
    periods=48, freq='h'  # lowercase h to avoid deprecation warning
)

print("\nNext 48-hour forecast (head):")
print(forecast_df.head())

# Optionally save:
# forecast_df.to_csv("forecast_hybrid_48.csv")
