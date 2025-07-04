# Model Comparison

# 1. Rolling Mean Naive Forecast
# - Uses a rolling window mean for predictions.
# - Simple, fast, but does not capture seasonality or complex dependencies.

# 2. SARIMA (Seasonal ARIMA)
# - Captures seasonality and trends in the data.
# - More robust for time series with daily/weekly cycles.

# Example: Compare RMSE for both models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np

def rolling_mean_forecast(train, test, window=24):
    mean_val = train[-window:].mean()
    return np.full_like(test, mean_val, dtype=np.float64)

def sarima_forecast(train, test, order=(1,1,1), seasonal_order=(1,1,1,24)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    fit = model.fit(disp=False)
    return fit.forecast(steps=len(test))

# Example usage for one column (e.g., 'CO(GT)')
train = df_clean['CO(GT)'].iloc[:-test_size]
test = df_clean['CO(GT)'].iloc[-test_size:]

# Rolling mean
y_pred_rm = rolling_mean_forecast(train, test, window=24)
rmse_rm = np.sqrt(mean_squared_error(test, y_pred_rm))

# SARIMA
y_pred_sarima = sarima_forecast(train, test)
rmse_sarima = np.sqrt(mean_squared_error(test, y_pred_sarima))

print(f"Rolling Mean RMSE: {rmse_rm}")
print(f"SARIMA RMSE: {rmse_sarima}")

# Feature Importance

# For tree-based models (e.g., XGBoost or RandomForest)
importances = model.feature_importances_
feature_names = X_train.columns
important_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print("Top features:")
for feat, imp in important_features[:10]:
    print(f"{feat}: {imp}")

# For linear models
coefs = model.coef_
for feat, coef in zip(feature_names, coefs):
    print(f"{feat}: {coef}")

# Residual Analysis

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

residuals = test - y_pred_sarima
plt.figure(figsize=(10,4))
plt.plot(residuals)
plt.title("Residuals of Final Model")
plt.show()

plot_acf(residuals.dropna(), lags=48)
plt.title("ACF of Residuals")
plt.show()

# Error Analysis

# Identify periods with largest errors
errors = np.abs(test - y_pred_sarima)
worst_indices = errors.sort_values(ascending=False).index[:5]
print("Worst prediction periods:")
for idx in worst_indices:
    print(f"{idx}: True={test.loc[idx]}, Pred={y_pred_sarima.loc[idx]}, Error={errors.loc[idx]}")

# Feature Engineering

# Example: Add lag and rolling features
for col in REQUIRED_COLUMNS:
    df_clean[f"{col}_lag_24"] = df_clean[col].shift(24)
    df_clean[f"{col}_rolling_24h_mean"] = df_clean[col].rolling(24).mean()

# Example: Add interaction feature
df_clean['temp_humidity_interaction'] = df_clean['T'] * df_clean['RH']

# Evaluate impact
# (Re-train model and compare RMSE before and after adding features)
