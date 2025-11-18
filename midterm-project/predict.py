from typing import Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Input data model for water potability record
class WaterPotabilityRecord(BaseModel):
    ph: Optional[float] = Field(None, ge=0.0)
    Hardness: Optional[float] = Field(None, ge=0.0)
    Solids: Optional[float] = Field(None, ge=0.0)
    Chloramines: Optional[float] = Field(None, ge=0.0)
    Sulfate: Optional[float] = Field(None, ge=0.0)
    Conductivity: Optional[float] = Field(None, ge=0.0)
    Organic_carbon: Optional[float] = Field(None, ge=0.0)
    Trihalomethanes: Optional[float] = Field(None, ge=0.0)
    Turbidity: Optional[float] = Field(None, ge=0.0)


# Output data model for potability prediction response
class PotabilityPredictionResponse(BaseModel):
    potability_probability: float
    is_potable: bool


# load the trained model from a joblib file
model = joblib.load("water_potability_prediction_model_v1_0.joblib")


# function to predict potability probability
def predict_potability(record: WaterPotabilityRecord) -> float:
    data = pd.DataFrame([record.model_dump()])
    prob = model.predict_proba(data)[0, 1]
    return float(prob)


# create FastAPI app
app = FastAPI(title="water-potability-prediction")


# define a GET / endpoint that returns a welcome message
@app.get("/")
def home():
    return {"message": "Welcome to the Water Potability Prediction API!"}


# define a POST /predict endpoint that takes a WaterPotabilityRecord and returns PotabilityPredictionResponse
@app.post("/predict")
def predict(record: WaterPotabilityRecord) -> PotabilityPredictionResponse:
    prob = predict_potability(record)
    return PotabilityPredictionResponse(
        potability_probability=prob,
        is_potable=prob >= 0.5
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
