# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and feature list
model = joblib.load("house_price_model.pkl")
feature_names = joblib.load("house_features.pkl")

app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using ML model"
)

# Input schema
class PredictionInput(BaseModel):
    area: float
    bedrooms: float
    bathrooms: float

# Output schema
class PredictionOutput(BaseModel):
    prediction: float

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "House Price Prediction API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Create DataFrame with correct feature names
        features = pd.DataFrame([[input_data.area, input_data.bedrooms, input_data.bathrooms]],
                                columns=feature_names)
        # Make prediction
        prediction = model.predict(features)[0]
        return PredictionOutput(prediction=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Model information endpoint
@app.get("/model-info")
def model_info():
    return {
        "model_type": "LinearRegression",
        "problem_type": "regression",
        "features": feature_names
    }
