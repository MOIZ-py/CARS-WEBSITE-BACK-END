import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---------------- CONFIG ----------------
DATA_FILE = "DATASET.csv"
MODEL_OUT = "car_price_model.joblib"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_FILE)

# Ensure correct column names
df.columns = df.columns.str.lower()

# Select required columns
df = df[["brand", "cartype", "enginesize", "fueltype", "price","status"]]

# Convert numeric
df["enginesize"] = pd.to_numeric(df["enginesize"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

df.dropna(inplace=True)

print(f"Dataset ready: {len(df)} rows")

# ---------------- FEATURES ----------------
X = df.drop("price", axis=1)
y = df["price"]

categorical_features = ["brand", "cartype", "fueltype","status"]
numerical_features = ["enginesize"]

# ---------------- PIPELINE ----------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numerical_features)
])

model = Pipeline([
    ("prep", preprocessor),
    ("regressor", Ridge(alpha=10))
])

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nModel Performance")
print(f"MAE: {mae:,.2f}")
print(f"R²: {r2:.4f}")

# ---------------- SAVE ----------------
joblib.dump(model, MODEL_OUT)
print(f"\nModel saved as {MODEL_OUT}")

# ---------------- USER INPUT PREDICTION ----------------
sample_car = pd.DataFrame({
    "brand": ["Toyota"],
    "cartype": ["SUV"],
    "fueltype": ["Petrol"],
    "enginesize": [3.0],
    "status" : ["used"]

})

predicted_price = model.predict(sample_car)[0]
print(f"\nPredicted Price: ${predicted_price:,.2f}")
