from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

app = FastAPI()
class LeadData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

class ConvertResponse(BaseModel):
    conversion_probability: float
    is_converted: bool

def predict_single(lead_data):
    result = pipeline.predict_proba(lead_data)[0, 1]
    return float(result)


@app.post("/predict")
def predict(lead_data: LeadData) -> ConvertResponse:
    conversion_probability = predict_single(lead_data.model_dump())
    is_converted = conversion_probability >= 0.5
    return ConvertResponse(
        conversion_probability=conversion_probability,
        is_converted=is_converted
    )
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
