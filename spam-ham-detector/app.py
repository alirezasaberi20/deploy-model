from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the pre-trained model and vectorizer
model = joblib.load("classifier.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Define the FastAPI app
app = FastAPI()

# Define the request schema
class TextData(BaseModel):
    text: str

@app.get("/")
def read_root():
    """
    Root endpoint to verify the API is running.
    """
    return {"message": "Welcome to the Spam/Ham Classifier API"}

@app.post("/predict/")
def predict(data: TextData):
    """
    Predict whether the input text is spam or ham.
    """
    # Extract text from the request
    input_text = data.text

    # Ensure input is valid
    if not input_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    # Vectorize the input text
    vectorized_text = vectorizer.transform([input_text])
    
    # Predict using the model
    prediction = model.predict(vectorized_text)[0]  # Returns "ham" or "spam"
    
    return {"text": input_text, "prediction": prediction}
