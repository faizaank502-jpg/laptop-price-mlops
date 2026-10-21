# Step 12: Training script – src/train.py
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

SEED = 42  # fixed seed = reproducible results

print("Loading dataset")
df = pd.read_csv("data/processed/laptops_clean.csv")
X = df.drop(columns=["Price"])
y = np.log(df["Price"])

cat_cols = ["Company", "TypeName", "Cpu_brand", "Gpu_brand", "OS"]
num_cols = [c for c in X.columns if c not in cat_cols]

print("Preprocessing data")
prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
}

print("Training ML models")
results, best_name, best_score, best_pipe = {}, None, -1e9, None
for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(np.exp(y_test), np.exp(pred))
    results[name] = {"r2": round(r2, 4), "mae_price": round(mae, 2)}
    print(f"{name:17s} R2 = {r2:.4f}   MAE (price) = {mae:,.2f}")
    if r2 > best_score:
        best_name, best_score, best_pipe = name, r2, pipe

joblib.dump(best_pipe, "models/laptop_price_model.joblib")
with open("reports/metrics.json", "w") as f:
    json.dump({"seed": SEED, "best_model": best_name, "results": results}, f, indent=2)

print(f"\nBest model: {best_name} -> saved to models/laptop_price_model.joblib")
