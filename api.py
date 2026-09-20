from fastapi import FastAPI

from predict_anomaly import predict_anomaly

app = FastAPI()

@app.get("/")
def home():

    return {
        "status": "SmartAgri API Running"
    }

@app.post("/predict")
def predict(payload: dict):

    result = predict_anomaly(payload)

    return result