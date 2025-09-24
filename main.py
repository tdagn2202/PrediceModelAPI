import os
import sys
import logging
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting FastAPI app...")

# Load model, scaler, and columns safely
try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), "model.pkl"))
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), "scaler.pkl"))
    columns = joblib.load(os.path.join(os.path.dirname(__file__), "columns.pkl"))
    if "Year" in columns:
        columns.remove("Year")
    logger.info("Model, scaler, and columns loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model/scaler/columns: {e}")
    sys.exit(1)

# FastAPI app
app = FastAPI(title="Crop Yield Prediction API")

# Input schema
class YieldInput(BaseModel):
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float
    Area: str
    Item: str

@app.get("/")
def root():
    return {"message": "Crop Yield Prediction API is running. Go to /docs for Swagger UI."}

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/predice")
def predict_yield(data: YieldInput):
    # Build one-row DataFrame
    df_input = pd.DataFrame([{
        "average_rain_fall_mm_per_year": data.average_rain_fall_mm_per_year,
        "pesticides_tonnes": data.pesticides_tonnes,
        "avg_temp": data.avg_temp,
        "Area": data.Area,
        "Item": data.Item
    }])

    # One-hot encode
    df_input = pd.get_dummies(df_input, columns=["Area", "Item"], prefix=["Country", "Item"])

    # Align with training columns
    df_input = df_input.reindex(columns=columns, fill_value=0)

    # Scale
    X = scaler.transform(df_input)

    # Predict
    prediction = model.predict(X)

    return {"predicted_yield": float(prediction[0])}

# Start the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
