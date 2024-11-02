from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn

# Initialize the FastAPI app
app = FastAPI(title="Sepsis Prediction API")

# Load the model and scaler
def load_model_and_scaler():
    with open("model.pkl", "rb") as f1, open("scaler.pkl", "rb") as f2:
        return pickle.load(f1), pickle.load(f2)

model, scaler = load_model_and_scaler()

def predict(df):
    # Scaling and prediction logic
    scaled_df = scaler.transform(df)
    predicted_labels = model.predict(scaled_df)
    probabilities = model.predict_proba(scaled_df).max(axis=1)

    response = []
    for label, proba in zip(predicted_labels, probabilities):
        output = {
            "prediction": "Patient has sepsis" if label == 1 else "Patient does not have sepsis",
            "probability of prediction": f"{round(proba * 100)}%"
        }
        response.append(output)

    return response

class Patient(BaseModel):
    Plasma_glucose: int
    Blood_Work_R1: int
    Blood_Pressure: int
    Blood_Work_R3: float
    BMI: float
    Blood_Work_R4: float
    Patient_age: int

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict")
def predict_sepsis(patient: Patient):
    data = pd.DataFrame(patient.dict(), index=[0])
    parsed = predict(df=data)
    return {"output": parsed}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
