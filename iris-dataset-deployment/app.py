from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris  # Import iris dataset for class names

app = FastAPI()

# Load the trained model and iris dataset
model = joblib.load('model.joblib')
iris = load_iris()

# Define input data model
class PredictionInput(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post("/predict/")
def predict(input_data: PredictionInput):
    try:
        features = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(features)
        return {
            "class": int(prediction[0]),
            "class_name": iris.target_names[prediction[0]]
        }
    except Exception as e:
        return {"error": str(e)}