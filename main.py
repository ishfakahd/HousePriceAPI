# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import logging

# -------------------- Logging Setup --------------------
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------- Load Model --------------------
model = joblib.load("house_price_model.pkl")
features = joblib.load("house_features.pkl")

# Calculate residual std for confidence scores
import pandas as pd
df = pd.read_csv("housing.csv")
X_train = df[features]
y_train = df['price']
residuals = y_train - model.predict(X_train)
residual_std = np.std(residuals)

# -------------------- FastAPI Setup --------------------
app = FastAPI(
    title="House Price Prediction API",
    description="API to predict house prices using ML model",
    version="0.1.0"
)

# -------------------- Pydantic Models --------------------
class PredictionInput(BaseModel):
    area: float
    bedrooms: float
    bathrooms: float

class PredictionOutput(BaseModel):
    prediction: float
    confidence: float = None

class BatchPredictionInput(BaseModel):
    inputs: List[PredictionInput]

# -------------------- API Endpoints --------------------
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "House Price Prediction API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        features_array = np.array([[input_data.area, input_data.bedrooms, input_data.bathrooms]])
        prediction = model.predict(features_array)[0]
        confidence = 1.96 * residual_std  # 95% confidence interval
        logging.info(f"Prediction: {prediction}, Confidence: {confidence}, Input: {input_data.dict()}")
        return PredictionOutput(prediction=prediction, confidence=float(confidence))
    except Exception as e:
        logging.error(f"Error: {e}, Input: {input_data.dict()}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-batch")
def predict_batch(batch_input: BatchPredictionInput):
    try:
        predictions = []
        for item in batch_input.inputs:
            features_array = np.array([[item.area, item.bedrooms, item.bathrooms]])
            pred = model.predict(features_array)[0]
            confidence = 1.96 * residual_std
            predictions.append({
                "input": item.dict(),
                "prediction": float(pred),
                "confidence": float(confidence)
            })
            logging.info(f"Batch Prediction: {pred}, Confidence: {confidence}, Input: {item.dict()}")
        return {"predictions": predictions}
    except Exception as e:
        logging.error(f"Batch Prediction Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "LinearRegression",
        "problem_type": "regression",
        "features": features
    }
