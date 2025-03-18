import pandas as pd
import xgboost as xgb

# Load dataset
df = pd.read_csv("crop_production_cleaned.csv")

# ✅ Instead of printing everything, show only the first 5 rows
print("✅ CSV Loaded Successfully! Showing first 5 rows:")
print(df.head())  # Shows first 5 rows only

# Create 'Yield' column (Production per Area)
df["Yield"] = df["Production"] / df["Area"]

# Drop unnecessary columns
columns_to_drop = ["State_Name", "District_Name"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

# One-hot encode categorical columns (Season, Crop) and clean column names
df = pd.get_dummies(df, columns=["Season", "Crop"], drop_first=True)
df.columns = df.columns.str.strip()  # Remove extra spaces

# Define features (X) and target variable (y)
X = df.drop(["Production", "Yield"], axis=1)
y = df["Yield"]

# Train the XGBoost model
model = xgb.XGBRegressor()
model.fit(X, y)

# Save the trained model and feature names
model.save_model("model.json")
X.columns.to_series().to_csv("model_features.csv", index=False)

print("✅ Model training complete! Files saved: model.json, model_features.csv")
