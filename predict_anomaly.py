
import pandas as pd
import numpy as np
import joblib
from diseasse.predict import predict_disease
from recommendation_engine import generate_recommendation
from pathlib import Path


# ==========================
# LOAD TRAINED FILES
# ==========================

model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")


# ==========================
# MAIN FUNCTION
# ==========================

def predict_anomaly(payload):

    # ==========================
    # EXTRACT FEATURES
    # ==========================

    row = {
        "soil_moisture":
            payload["data"]["soil"]["soil_moisture"],

        "ph":
            payload["data"]["soil"]["ph"],

        "nitrogen":
            payload["data"]["soil"]["npk"]["nitrogen"],

        "phosphorus":
            payload["data"]["soil"]["npk"]["phosphorus"],

        "potassium":
            payload["data"]["soil"]["npk"]["potassium"],

        "temperature":
            payload["data"]["weather"]["temperature"],

        "humidity":
            payload["data"]["weather"]["humidity"],

        "wind_speed":
            payload["data"]["weather"]["wind_speed"],

        "period":
            payload["data"]["weather"]["period"],

        "sky":
            payload["data"]["weather"]["sky"],

        "precipitation":
            payload["data"]["weather"]["precipitation"],

        "temperature_condition":
            payload["data"]["weather"]["temperature_condition"],

        "wind_condition":
            payload["data"]["weather"]["wind_condition"]
    }

    # ==========================
    # DATAFRAME
    # ==========================

    df = pd.DataFrame([row])

    # ==========================
    # ONE HOT ENCODING
    # ==========================

    df = pd.get_dummies(df)

    # ==========================
    # MATCH TRAINING FEATURES
    # ==========================

    for col in feature_names:

        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    # ==========================
    # SCALING
    # ==========================

    X_scaled = scaler.transform(df)

        # ==========================
    # PREDICTION
    # ==========================

    prediction = int(model.predict(X_scaled)[0])

    score = float(model.decision_function(X_scaled)[0])

    anomaly_detected = bool(prediction == -1)

    # ==========================
    # DISEASE PREDICTION
    # ==========================


    BASE_DIR = Path(__file__).resolve().parent

    IMAGE_PATH = BASE_DIR / "diseasse" / "image_test.jpg"

    disease_result = predict_disease(str(IMAGE_PATH))

    disease_detected = disease_result["detected"]

    disease_name = disease_result["name"]

    disease_confidence = disease_result["confidence"]

    disease_severity = disease_result["severity"]

    disease_action = disease_result["action"]


    # ==========================
    # ANOMALY SEVERITY
    # ==========================

    if not anomaly_detected:
        severity = "none"
    else:

        if score <= -0.03:
            severity = "critical"

        elif score <= -0.015:
            severity = "high"

        elif score <= 0:
            severity = "medium"

        else:
            severity = "low"


    # ==========================
    # DISEASE PREDICTION
    # ==========================

    disease_result = predict_disease(
        r"C:\Users\A\Desktop\smartAgri AI 2\diseasse\image_test.jpg"
    )

    disease_detected = disease_result["detected"]

    disease_name = disease_result["name"]

    disease_confidence = disease_result["confidence"]

    disease_severity = disease_result["severity"]

    disease_action = disease_result["action"]

   
    if disease_detected:

        if disease_severity == "High":
            health_status = "unhealthy"

        elif disease_severity == "Medium":
            health_status = "warning"

        else:
            health_status = "healthy"

    elif anomaly_detected:

        if severity in ["critical", "high"]:
            health_status = "unhealthy"

        else:
            health_status = "warning"

    else:

        if (
            60 <= row["soil_moisture"] <= 80
            and 18 <= row["temperature"] <= 30
            and 60 <= row["humidity"] <= 80
            and 6.0 <= row["ph"] <= 6.8
        ):
            health_status = "healthy"
        else:
            health_status = "warning"

##### REASONS 

    reasons = []

    if row["soil_moisture"] < 30:
        reasons.append("Low soil moisture")

    if row["temperature"] > 35:
        reasons.append("High temperature")

    if row["humidity"] < 40:
        reasons.append("Low humidity")

    if row["ph"] < 5.5:
        reasons.append("Low soil pH")

    if row["nitrogen"] < 40:
        reasons.append("Nitrogen deficiency")

    reason = ", ".join(reasons)

    if not reason:
        reason = None
 
    # ==========================
    # RECOMMENDATIONS
    # ==========================

    recommendations = generate_recommendation(row)

    # ==========================
    # FINAL RECOMMENDATION
    # ==========================

    if disease_detected:

        recommendation_action = disease_action

        recommendation_priority = (
            disease_severity.lower()
        )

        recommendation_message = (
            f"{disease_name} detected."
        )

    else:

        recommendation_action = recommendations[0]["action"]

        recommendation_priority = (
            recommendations[0]["priority"]
        )

        recommendation_message = reason

    # ==========================
    # IRRIGATION LOGIC
    # ==========================

    irrigation_needed = row["soil_moisture"] < 30

    water_amount = (
        4.5
        if irrigation_needed
        else 0
    )

    # ==========================
    # RESPONSE
    # ==========================
    response = {

        "success": True,

        "health": {
            "status": health_status
        },

        "anomaly": {
            "detected": anomaly_detected,
            "severity": severity,
            "reason": reason
        },

        "disease": {
            "detected": disease_detected,
            "name": disease_name,
            "confidence": round(disease_confidence, 2)
        },

        "irrigation": {
            "recommended": irrigation_needed,
            "water_amount": water_amount,
            "unit": "L/m2"
        },

        "recommendation": {
            "action": recommendation_action,
            "priority": recommendation_priority,
            "message": recommendation_message
        }
    }
    return response


# ==========================
# TEST
# ==========================

if __name__ == "__main__":

    test_payload = {
        "field_id": 1,
        "sector_id": 1,
        "node_id": "A84F92",

        "data": {
            "soil": {
                "soil_moisture": 5,
                "ph": 3.2,
                "npk": {
                    "nitrogen": 0,
                    "phosphorus": 8,
                    "potassium": 3
                }
            },

            "weather": {
                "period": "day",
                "sky": "clear",
                "precipitation": "none",
                "temperature": 49,
                "temperature_condition": "hot",
                "humidity": 5,
                "wind_speed": 42,
                "wind_condition": "moderate"
            }
        }
    }

    result = predict_anomaly(test_payload)

    import json

    print(
        json.dumps(
            result,
            indent=4
        )
    )
