# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset by reading the CSV file, cleaning column names, sorting by time, and filling missing values using interpolation.
2. Perform feature engineering by creating time-based features (hour_sin, hour_cos) and lag features for temperature, PM2.5, and solar radiation.
3. Split the processed data into training and testing sets (80% training and 20% testing) and select important features for the model.
4. Train a Random Forest Regressor, evaluate the model using R² and MAE metrics, and visualize actual vs predicted values and feature importance using plots.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: B.PRAVEEN RAJ 
RegisterNumber:  25019206
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================================================================
# 1. LOAD & CLEAN DATA
# ==============================================================================
# Read raw CSV and fix column whitespace
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()

# 1.1 Chronological Sorting: Ensure time flows correctly
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# 1.2 Interpolation: Fill gaps (up to 10 rows) to keep the timeline continuous
cols_to_fill = ['tem', 'pm2_5', 'tsr', 'hum', 'pressure', 'wind_speed', 'illumination', 'co2']
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear', limit=10)

# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
# 2.1 Cyclical Time Features: Convert hour into circle coordinates (Sin/Cos)
df['hour'] = df['time'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# 2.2 Lag Features: Give the model 'Memory' of what happened 1 and 2 steps ago
targets = ['tem', 'pm2_5', 'tsr']
for t in targets:
    df[f'{t}_lag1'] = df[t].shift(1)
    df[f'{t}_lag2'] = df[t].shift(2)

# 2.3 Cleanup: Drop rows where lags are NaN and save processed data
processed_df = df.dropna(subset=['tem_lag2', 'pm2_5_lag2', 'tsr_lag2', 'hum', 'pressure']).reset_index(drop=True)
processed_df.to_csv("combined_processed_weather_data.csv", index=False)

# Define the final high-performance feature set
features = [
    'hum', 'pressure', 'wind_speed', 'illumination', 'co2',
    'hour_sin', 'hour_cos', 'tem_lag1', 'pm2_5_lag1', 'tsr_lag1'
]
# Print summary of feature engineering
print("--- Feature Engineering Summary ---")
print(f"Original rows: {len(df)}")
print(f"Processed rows (after lags/cleaning): {len(processed_df)}")
print(f"Final high-performance feature set:",features)
# ==============================================================================
# 3. TRAIN-TEST SPLIT (Chronological)
# ==============================================================================
# Take the first 80% for training and the final 20% for testing (no shuffling)
split_idx = int(len(processed_df) * 0.8)
train, test = processed_df.iloc[:split_idx], processed_df.iloc[split_idx:]
X_train, X_test = train[features], test[features]

models = {}
results = {}

# ==============================================================================
# 4. TRAINING & PERFORMANCE EVALUATION
# ==============================================================================
target_meta = {
    'tem': ('Temperature', '°C', 'red'),
    'pm2_5': ('Pollution (PM2.5)', 'µg/m³', 'green'),
    'tsr': ('Energy (Solar Radiation)', 'W/m²', 'orange')
}

for target in targets:
    y_train, y_test = train[target], test[target]
    
    # Random Forest with high-depth logic for complex weather patterns
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    models[target] = model
    
    # Store metrics for interpretation
    results[target] = {
        'r2': r2_score(y_test, preds),
        'mae': mean_absolute_error(y_test, preds),
        'preds': preds,
        'actual': y_test.values
    }

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for i, target in enumerate(targets):
    label, unit, color = target_meta[target]
    res = results[target]
    
    # Plot 1: Actual vs Predicted (Showing the last 150 points for detail)
    axes[i, 0].plot(res['actual'][-150:], label='Actual', color='black', alpha=0.4, linewidth=2)
    axes[i, 0].plot(res['preds'][-150:], label='Predicted', color=color, linestyle='--', linewidth=2)
    axes[i, 0].set_title(f"{label}: Actual vs Predicted\n$R^2$: {res['r2']:.3f} | MAE: {res['mae']:.2f}")
    axes[i, 0].set_ylabel(unit)
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    
    # Plot 2: Feature Importance (Which sensors influenced this target most?)
    importances = pd.Series(models[target].feature_importances_, index=features).sort_values()
    importances.plot(kind='barh', ax=axes[i, 1], color=color, alpha=0.7)
    axes[i, 1].set_title(f"Key Drivers: {label}")

plt.tight_layout()
plt.show()

*/
```

## Output:
<img width="1092" height="393" alt="Screenshot 2026-03-17 082338" src="https://github.com/user-attachments/assets/14ce7bb7-1862-4d49-8100-d36cf40fe5fa" />
<img width="1088" height="379" alt="Screenshot 2026-03-17 082353" src="https://github.com/user-attachments/assets/0cc6d16c-0960-4c15-ba31-b41ad9a5aaf6" />
<img width="1055" height="386" alt="Screenshot 2026-03-17 082422" src="https://github.com/user-attachments/assets/9702a041-92ef-41c7-a943-971c4500dd0f" />



## Result:
Thus the program to Implement of Random Forest Algorithm for Weather Prediction is written and verified using python programming.
