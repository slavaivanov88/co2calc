import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0  # avoid division by zero
    if not np.any(non_zero):
        return np.nan  # return NaN if all actual values are zero
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

# Load data from CSV
csv_path = 'co2.csv'
df = pd.read_csv(csv_path)

# Add new manual rows with missing CO2 values 
new_rows = pd.DataFrame([
    {'Type': 'Pipe', 'Material': 'Concrete', 'Diameter': 300, 'Class': 'B', 'Description': 'round', 'CO2ekv kg/m': np.nan},
    {'Type': 'Pipe', 'Material': 'Concrete', 'Diameter': 600, 'Class': 'Dr', 'Description': 'round', 'CO2ekv kg/m': np.nan},
    {'Type': 'Pipe', 'Material': 'Concrete', 'Diameter': 800, 'Class': 'Br', 'Description': 'footed', 'CO2ekv kg/m': np.nan},
    {'Type': 'Pipe', 'Material': 'Concrete', 'Diameter': 1200, 'Class': 'Dr', 'Description': 'footed', 'CO2ekv kg/m': np.nan},
    {'Type': 'Pipe', 'Material': 'Concrete', 'Diameter': 1600, 'Class': 'Dr', 'Description': 'chamfered', 'CO2ekv kg/m': np.nan}
])

# Append new rows
full_df = pd.concat([df, new_rows], ignore_index=True)

# Rename columns to match expected names
full_df = full_df.rename(columns={
    'Type': 'type',
    'Material': 'material',
    'Diameter': 'diameter_mm',
    'Class': 'class',
    'Description': 'description',
    'CO2ekv kg/m': 'co2_emission'
})

# Sort original table by diameter
print("\nOriginal Table (sorted by diameter):")
print(full_df.sort_values(by='diameter_mm').head(50))

# Split into labeled and missing co2_emission
df_labeled = full_df[full_df['co2_emission'].notna()].copy()
df_missing = full_df[full_df['co2_emission'].isna()].copy()

print(f"\nLabeled entries: {len(df_labeled)}")
print(f"Missing entries: {len(df_missing)}")

# Combine for consistent encoding
combined = pd.concat([df_labeled, df_missing], ignore_index=True)

# One-hot encoding for class and description
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_features = combined[['class', 'description']].fillna('')
ohe_encoded = ohe.fit_transform(categorical_features)

# Numerical features
numerical_feature = combined[['diameter_mm']].fillna(0).values

# Combine all features
X_all = np.hstack([numerical_feature, ohe_encoded])

# Re-split features into train and missing
X_train = X_all[:len(df_labeled)]
y_train = df_labeled['co2_emission'].values
X_missing = X_all[len(df_labeled):]

# Train model
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# Predict and calculate confidence
df_missing['predicted_co2_emission'] = reg.predict(X_missing)
all_preds = np.stack([tree.predict(X_missing) for tree in reg.estimators_])
df_missing['confidence'] = all_preds.std(axis=0)

# Show predictions
print("\nMissing elements with predicted CO2 emissions:")
print(df_missing[['type', 'material', 'diameter_mm', 'class', 'description', 'co2_emission', 'predicted_co2_emission', 'confidence']])

# --- Manual Input of 5 Known Values ---
# Here, you can input the 5 actual CO2 values for comparison against the predictions.
print("\nPlease manually input 5 rows of known CO2 values for comparison:")

known_values = [
    {'type': 'Pipe', 'material': 'Concrete', 'diameter_mm': 300, 'class': 'B', 'description': 'round', 'actual_co2': 26.7},
    {'type': 'Pipe', 'material': 'Concrete', 'diameter_mm': 600, 'class': 'Dr', 'description': 'round', 'actual_co2': 68.4},
    {'type': 'Pipe', 'material': 'Concrete', 'diameter_mm': 800, 'class': 'Br', 'description': 'footed', 'actual_co2': 141.3},
    {'type': 'Pipe', 'material': 'Concrete', 'diameter_mm': 1200, 'class': 'Dr', 'description': 'footed', 'actual_co2': 285.0},
    {'type': 'Pipe', 'material': 'Concrete', 'diameter_mm': 1600, 'class': 'Dr', 'description': 'chamfered', 'actual_co2': 428.0}
]

# Convert the known values to a DataFrame
df_known = pd.DataFrame(known_values)

# Manually match the known values with the predicted values
X_known = np.hstack([
    df_known[['diameter_mm']].values,
    ohe.transform(df_known[['class', 'description']].fillna(''))
])

# Get predictions for the known values
predictions_known = reg.predict(X_known)

# Comparison between known and predicted values
df_known['predicted_co2'] = predictions_known

# Calculate performance metrics for the comparison
mae = mean_absolute_error(df_known['actual_co2'], df_known['predicted_co2'])
mse = mean_squared_error(df_known['actual_co2'], df_known['predicted_co2'])
rmse = sqrt(mse)
r2 = r2_score(df_known['actual_co2'], df_known['predicted_co2'])

y_true = df_known['actual_co2']
y_pred = df_known['predicted_co2']
# Existing metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

# NEW: MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)

# Print performance metrics
print("\nComparison of Predicted vs Known CO2 Emissions:")
print(df_known[['type', 'material', 'diameter_mm', 'class', 'description', 'actual_co2', 'predicted_co2']])

# Print all metrics
print(f"\n--- Prediction Quality Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# --- Visualize Feature Importances ---
# Get feature names
encoded_feature_names = ['diameter_mm'] + list(ohe.get_feature_names_out(['class', 'description']))

# Get importance values from trained model
importances = reg.feature_importances_

# Create a DataFrame for easy sorting and display
feature_importance_df = pd.DataFrame({
    'feature': encoded_feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Print the top feature importances
print("\nTop Feature Importances:")
print(feature_importance_df.head(10))



# --- Visualizations ---

# Scatter plot: Actual vs Predicted for known values
plt.figure(figsize=(8, 6))
plt.scatter(df_known['actual_co2'], df_known['predicted_co2'], alpha=0.7, color='orange')
plt.plot([min(df_known['actual_co2']), max(df_known['actual_co2'])], [min(df_known['actual_co2']), max(df_known['actual_co2'])], color='red', linestyle='--', lw=2)
plt.title('Actual vs Predicted CO2 Emissions (Known Values)')
plt.xlabel('Actual CO2 Emissions (kg CO2e)')
plt.ylabel('Predicted CO2 Emissions (kg CO2e)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals plot for known values
residuals_known = df_known['actual_co2'] - df_known['predicted_co2']
plt.figure(figsize=(8, 6))
plt.scatter(df_known['actual_co2'], residuals_known, alpha=0.7, color='purple')
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.title('Residuals vs Actual CO2 Emissions (Known Values)')
plt.xlabel('Actual CO2 Emissions (kg CO2e)')
plt.ylabel('Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

# Prediction error distribution for known values
plt.figure(figsize=(8, 6))
plt.hist(residuals_known, bins=30, edgecolor='black', color='lightblue')
plt.title('Distribution of Prediction Errors (Residuals)')
plt.xlabel('Prediction Error (Residuals)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

