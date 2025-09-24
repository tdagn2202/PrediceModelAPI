
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model, scaler, và danh sách cột
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

columns = joblib.load("columns.pkl")
if "Year" in columns:
    columns.remove("Year")


app = FastAPI()

class YieldInput(BaseModel):
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float
    Area: str
    Item: str

@app.get("/")
def root():
    return {"message": "Crop Yield Prediction API is running. Go to /docs for Swagger UI."}

@app.post("/predice")
def predict_yield(data: YieldInput):
    # Build one row DataFrame
    df_input = pd.DataFrame([{
        "average_rain_fall_mm_per_year": data.average_rain_fall_mm_per_year,
        "pesticides_tonnes": data.pesticides_tonnes,
        "avg_temp": data.avg_temp,
        "Area": data.Area,
        "Item": data.Item
    }])

    # ✅ Use the same prefix as training
    df_input = pd.get_dummies(df_input, columns=["Area", "Item"], prefix=["Country", "Item"])

    # ✅ Align with training columns
    df_input = df_input.reindex(columns=columns, fill_value=0)

    # Scale features
    X = scaler.transform(df_input)

    # Predict
    prediction = model.predict(X)

    return {"predicted_yield": float(prediction[0])}


if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8000))  # Railway injects PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)
