from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import io

app = FastAPI()

# Load model and scaler once at startup
model = joblib.load("aqi_forecast_model.pkl")
scaler = joblib.load("aqi_scaler.pkl")

# Features used during training
FEATURES = ['PM25', 'PM10', 'O3', 'SO2', 'NO', 'NO2', 'NOX', 'CO', 'CH4',
            'NMHC', 'THC', 'wind_speed', 'wind_gust_speed', 'wind_direction',
            'air_humidity', 'air_temperature', 'container_humidity',
            'container_temperature', 'solar_radiation']

INPUT_LEN = 24  # past 24 half-hour steps

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), parse_dates=["timestamp"])
        df = df.sort_values("timestamp")

        # Check if we have enough rows
        if len(df) < INPUT_LEN:
            raise ValueError("CSV must contain at least 24 rows for inference.")

        # Extract latest 24 records
        recent = df[FEATURES].iloc[-INPUT_LEN:]
        if recent.isnull().any().any():
            raise ValueError("Missing values found in required features.")

        x = recent.values.flatten().reshape(1, -1)
        x_scaled = scaler.transform(x)
        prediction = model.predict(x_scaled).flatten().tolist()

        return JSONResponse(content={
            "predicted_AQI_next_24h": prediction,
            "step": "Each step = 30 minutes",
            "unit": "AQI"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
