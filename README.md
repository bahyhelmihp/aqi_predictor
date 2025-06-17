# 🌫️ AQI Forecast Inference API

This repository provides a machine learning pipeline and API to predict the **next 24 hours of Air Quality Index (AQI)** using recent environmental sensor data. The model is trained on time-series data (every 30 minutes) and returns AQI predictions in one go.

---

## 🧠 How It Works

- 📊 Input: Last **12 hours of data** (24 steps × 30 minutes) from sensors
- 🎯 Output: AQI forecast for the next **24 hours** (48 values, each = 30 min)
- 🧱 Model: Tree-based regressor (LightGBM or XGBoost) wrapped in `MultiOutputRegressor`
- ⚡ Inference via FastAPI + CSV upload

---


### Start API Server
``uvicorn main:app --reload --port 8000``

### Request via cURL
``curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@recent_data.csv"
``

### Request via Python
```
import requests

files = {"file": open("recent_data.csv", "rb")}
res = requests.post("http://localhost:8000/predict", files=files)
print(res.json())
```

### Response
```
{
  "predicted_AQI_next_24h": [12.5, 13.2, ..., 16.1],
  "step": "Each step = 30 minutes",
  "unit": "AQI"
}
```
