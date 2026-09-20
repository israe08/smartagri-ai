import json

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==========================
# LOAD DATASET
# ==========================

DATASET_PATH = r"C:\Users\A\Desktop\smartAgri AI 2\ias42_tomato_ai_augmented_100000.json"
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    content = json.load(f)

data = content["data"]

df = pd.json_normalize(data)
df = df.rename(columns={
    "data.soil.soil_moisture": "soil_moisture",
    "data.soil.ph": "ph",
    "data.soil.npk.nitrogen": "nitrogen",
    "data.soil.npk.phosphorus": "phosphorus",
    "data.soil.npk.potassium": "potassium",
    "data.weather.temperature": "temperature",
    "data.weather.humidity": "humidity",
    "data.weather.wind_speed": "wind_speed",
    "data.weather.period": "period",
    "data.weather.sky": "sky",
    "data.weather.precipitation": "precipitation",
    "data.weather.temperature_condition": "temperature_condition",
    "data.weather.wind_condition": "wind_condition"
})



print(df.shape)
print(df.columns.tolist())

# ==========================
# NUMERIC FEATURES
# ==========================

numeric_features = [
    "soil_moisture",
    "ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "wind_speed"
]

# ==========================
# CATEGORICAL FEATURES
# ==========================

categorical_features = [
    "period",
    "sky",
    "precipitation",
    "temperature_condition",
    "wind_condition"
]

# ==========================
# CLEAN NUMERIC COLUMNS
# ==========================

for col in numeric_features:

    df[col] = (
        df[col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("°C", "", regex=False)
        .str.replace("km/h", "", regex=False)
        .str.strip()
    )

    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================
# HANDLE NULL VALUES
# ==========================

for col in numeric_features:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_features:
    df[col] = df[col].fillna("unknown")

# ==========================
# SELECT FEATURES
# ==========================

X = df[numeric_features + categorical_features].copy()

# ==========================
# ONE HOT ENCODING
# ==========================

X = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=False
)

print("Encoded Shape:", X.shape)

# ==========================
# SAVE FEATURE NAMES
# ==========================

feature_names = X.columns.tolist()

joblib.dump(
    feature_names,
    "feature_names.pkl"
)

print("feature_names.pkl saved")

# ==========================
# SCALING
# ==========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("scaler.pkl saved")

# ==========================
# TRAIN ISOLATION FOREST
# ==========================

model = IsolationForest(
    n_estimators=300,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)

print("Model Trained Successfully")

# ==========================
# SAVE MODEL
# ==========================

joblib.dump(
    model,
    "anomaly_model.pkl"
)

print("anomaly_model.pkl saved")

# ==========================
# TEST ON TRAINING DATA
# ==========================

predictions = model.predict(X_scaled)

df["anomaly"] = predictions

normal_count = len(df[df["anomaly"] == 1])
anomaly_count = len(df[df["anomaly"] == -1])

print("\n===== RESULTS =====")
print("Normal:", normal_count)
print("Anomalies:", anomaly_count)

# ==========================
# SAVE DETECTED ANOMALIES
# ==========================


anomalies_df = df[df["anomaly"] == -1]

anomalies_df.to_csv(
    "detected_anomalies.csv",
    index=False
)

print("detected_anomalies.csv saved")

scores = model.decision_function(X_scaled)

df["anomaly_score"] = scores

print("\nScore Statistics:")
print(df["anomaly_score"].describe())

print("\nMIN SCORE =", scores.min())
print("MAX SCORE =", scores.max())
print("MEAN SCORE =", scores.mean())
# ==========================
# EXTRA STATISTICS
# ==========================

preds = model.predict(X_scaled)

normal_count = int((preds == 1).sum())
anomaly_count = int((preds == -1).sum())

anomaly_rate = round(
    (anomaly_count / len(preds)) * 100,
    2
)

print("\n===== FINAL STATS =====")
print(f"Normal Records: {normal_count}")
print(f"Anomaly Records: {anomaly_count}")
print(f"Anomaly Rate: {anomaly_rate}%")

# ==========================
# SAVE SCORE LIMITS
# ==========================

score_config = {
    "min_score": float(scores.min()),
    "max_score": float(scores.max()),
    "mean_score": float(scores.mean())
}

joblib.dump(
    score_config,
    "score_config.pkl"
)

print("score_config.pkl saved")