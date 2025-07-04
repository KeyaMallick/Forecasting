
# Hybrid Forecasting for Air Quality Data

This script performs hybrid forecasting on the UCI Air Quality dataset using:
- Rolling mean naive forecast for stable columns.
- Last observed value for volatile columns (e.g., RH, NOx sensors).

## Features
- Preprocessing with forward and backward fill.
- Feature engineering (hour, day, etc.)
- Stationarity check (ADF) [optional].
- RMSE evaluation on last 10% of the dataset.
- Forecasting next 48 hours of air quality metrics.

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn

## Usage
1. Place the `AirQualityUCI.xlsx` in the same directory.
2. Run `hybrid_forecast_main.py`.

The script prints:
- RMSE on the last 10% of data for each variable.
- The 48-hour naive forecast values.

## Notes
- Columns with high volatility are handled using last-value prediction.
- Other columns use a rolling mean (window size = 5).
- Adjust the window size and volatile column list as needed.

