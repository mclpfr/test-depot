from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

api = FastAPI()

class PredictionResponse(BaseModel):
    prediction: int

# Charger le modèle
model = load("../models/rf_model_2023.joblib")

@app.post("/predict", response_model=PredictionResponse)
def predict():
    predicted_model = model.predict(input_data)
    return PredictionResponse(prediction=predicted_model)
