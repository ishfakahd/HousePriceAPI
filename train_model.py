import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("housing.csv")

# Only input features here
features = ['area', 'bedrooms', 'bathrooms']
X = df[features]
y = df['price']  # target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

r2_score = model.score(X_test, y_test)
print(f"Model R2 Score: {r2_score:.2f}")

# Save model and only input features
joblib.dump(model, "house_price_model.pkl")
joblib.dump(features, "house_features.pkl")
print("Model and features saved successfully!")
