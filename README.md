House Price Prediction API

Version 0.1.0

Project Overview

This project is a Machine Learning API built with FastAPI to predict house prices based on selected features.
It uses a Linear Regression model trained on the Kaggle Housing Prices dataset.

The API provides endpoints for inference, model information, and health checks.

Problem Description

Predict house prices using simple numerical features.


Input features:

area – House area in square feet

bedrooms – Number of bedrooms

bathrooms – Number of bathrooms

Output:

prediction – Estimated house price

Dataset: Kaggle Housing Prices Dataset

Model Choice

Model: Linear Regression

Reason: Simple regression model suitable for predicting continuous values like house prices.

Trained on the Kaggle dataset using area, bedrooms, and bathrooms as features.

Model evaluation: R² score on test data (printed during training).


Project Structure
HousePriceAPI/
├── main.py                # FastAPI application
├── train_model.py         # Script to train and save the ML model
├── housing.csv            # Dataset (Kaggle)
├── house_price_model.pkl  # Saved trained model
├── house_features.pkl     # Saved feature list
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

Setup & Installation

Clone the repository:

git clone <your-repo-url>
cd HousePriceAPI


Install dependencies:

pip install -r requirements.txt


Train the model (optional if house_price_model.pkl already exists):

python train_model.py


Run the FastAPI server:

python -m uvicorn main:app --reload


Open the API docs in your browser:

http://127.0.0.1:8000/docs

API Endpoints
1. Health Check

GET /

Returns API status.

Example Response:

{
  "status": "healthy",
  "message": "House Price Prediction API is running"
}

2. Predict House Price

POST /predict

Input: JSON with house features.

Request Body Example:

{
  "area": 2000,
  "bedrooms": 3,
  "bathrooms": 2
}


Response Example:

{
  "prediction": 450000.0
}

3. Model Information

GET /model-info

Returns model type, problem type, and feature names.

Response Example:

{
  "model_type": "LinearRegression",
  "problem_type": "regression",
  "features": ["area", "bedrooms", "bathrooms"]
}

Example Requests

Example 1

{
  "area": 1500,
  "bedrooms": 3,
  "bathrooms": 2
}


Example 2

{
  "area": 2500,
  "bedrooms": 4,
  "bathrooms": 3
}

Assumptions & Limitations

The dataset does not include the age of the house; only area, bedrooms, and bathrooms are used.

Linear Regression assumes a linear relationship between features and price.

Predictions may not generalize well to houses outside the dataset range.

Dependencies

Python 3.10+

FastAPI

Uvicorn

scikit-learn

pandas

numpy

joblib