# 🏠 House Price Prediction API

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

**Version:** 0.1.0  

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Problem Description](#problem-description)
3. [Model Choice](#model-choice)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [API Endpoints](#api-endpoints)
7. [Example Requests](#example-requests)
8. [Assumptions & Limitations](#assumptions--limitations)
9. [Dependencies](#dependencies)

---

## **Project Overview**

This project is a **Machine Learning API** built with **FastAPI** to predict house prices based on selected numerical features.  

- **Model:** Linear Regression  
- **Dataset:** [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)  
- **Functionality:** Provides API endpoints for:  
  1. Inference (predict house prices)  
  2. Model information  
  3. Health checks  

---

## **Problem Description**

The API predicts house prices using simple numerical features:

| Feature    | Description                     |
|------------|---------------------------------|
| area       | House area in square feet       |
| bedrooms   | Number of bedrooms             |
| bathrooms  | Number of bathrooms            |

**Output:**  
- `prediction` – Estimated house price

---

## **Model Choice**

- **Model:** Linear Regression  
- **Reason:** Simple regression model suitable for predicting continuous values like house prices.  
- **Features Used:** `area`, `bedrooms`, `bathrooms`  
- **Evaluation:** R² score printed during training  

---

## **Project Structure**

HousePriceAPI/
├── main.py # FastAPI application

├── train_model.py # Script to train and save the ML model

├── housing.csv # Dataset (Kaggle)

├── house_price_model.pkl # Saved trained model

├── house_features.pkl # Saved feature list

├── requirements.txt # Dependencies

└── README.md # Project documentation



---

## **Setup & Installation**

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd HousePriceAPI

2. Install dependencies
pip install -r requirements.txt

3. Train the model (optional if house_price_model.pkl already exists)
python train_model.py

4. Run the FastAPI server
python -m uvicorn main:app --reload

5. Open API docs in your browser
http://127.0.0.1:8000/docs

API Endpoints
1. Health Check

GET /

Returns API status

Example Response:

{
  "status": "healthy",
  "message": "House Price Prediction API is running"
}

2. Predict House Price

POST /predict

Input: JSON with house features

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

Returns model type, problem type, and feature names

Response Example:

{
  "model_type": "LinearRegression",
  "problem_type": "regression",
  "features": ["area", "bedrooms", "bathrooms"]
}

Example Requests

Example 1:

{
  "area": 1500,
  "bedrooms": 3,
  "bathrooms": 2
}


Example 2:

{
  "area": 2500,
  "bedrooms": 4,
  "bathrooms": 3
}
