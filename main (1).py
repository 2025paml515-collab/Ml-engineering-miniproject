# 1. Install dependencies
!pip install -q fastapi uvicorn pydantic nest_asyncio pyngrok requests joblib

import joblib
import numpy as np
import pandas as pd
from typing import Literal
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import nest_asyncio
import uvicorn
import threading
import time
import requests
import csv
import os

LOG_FILE = "prediction_logs.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "trip_distance", "passenger_count", "predicted_duration_seconds"])

def save_log(distance, passengers, predicted_time):
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), distance, passengers, predicted_time])
# 2. Save model artifacts
try:
    joblib.dump(rf_model, 'rf_model.joblib')
    joblib.dump(kmeans, 'kmeans.joblib')
    joblib.dump(pca, 'pca.joblib')
    joblib.dump(features, 'feature_columns.joblib')
    print('Artifacts saved successfully.')
except NameError:
    print('Models already loaded or defined.')

# 3. Feature engineering function
def build_features(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # Encode store_and_fwd_flag (N=0, Y=1)
    df['store_and_fwd_flag'] = (
        df['store_and_fwd_flag'].astype(str).str.upper().map({'N': 0, 'Y': 1}).fillna(0)
    )
    
    # Temporal features
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['is_weekend'] = df['pickup_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_minute'] = df['pickup_datetime'].dt.minute
    df['is_peak_hour'] = df['pickup_hour'].apply(lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 19) else 0)
    df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24.0)
    
    # Distance features
    df['distance_km'] = haversine_array(
        df['pickup_latitude'].values, df['pickup_longitude'].values,
        df['dropoff_latitude'].values, df['dropoff_longitude'].values
    )
    df['lat_diff'] = (df['dropoff_latitude'] - df['pickup_latitude']).abs()
    df['lon_diff'] = (df['dropoff_longitude'] - df['pickup_longitude']).abs()
    df['distance_manhattan_km'] = manhattan_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # Cluster + PCA features using the objects fit in M2
    df['pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']].values)
    df['dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']].values)
    pickup_pca = pca.transform(df[['pickup_latitude', 'pickup_longitude']].values)
    df['pca_num_1'] = pickup_pca[:, 0]
    df['pca_num_2'] = pickup_pca[:, 1]
    
    for col in features:
        if col not in df.columns:
            df[col] = 0
            
    return df[features].fillna(0)

# 4. Request/response schemas (Pydantic)
class TripRequest(BaseModel):
    vendor_id: int = Field(..., ge=1, le=2)
    pickup_datetime: datetime
    passenger_count: int = Field(..., ge=0, le=9)
    pickup_longitude: float = Field(..., ge=-180, le=180)
    pickup_latitude: float = Field(..., ge=-90, le=90)
    dropoff_longitude: float = Field(..., ge=-180, le=180)
    dropoff_latitude: float = Field(..., ge=-90, le=90)
    store_and_fwd_flag: Literal['N', 'Y', 'n', 'y'] = 'N'

    class Config:
        json_schema_extra = {
            'example': {
                'vendor_id': 2,
                'pickup_datetime': '2016-03-14T17:24:55',
                'passenger_count': 1,
                'pickup_longitude': -73.982155,
                'pickup_latitude': 40.767937,
                'dropoff_longitude': -73.964630,
                'dropoff_latitude': 40.765602,
                'store_and_fwd_flag': 'N'
            }
        }

class TripPredictionResponse(BaseModel):
    predicted_trip_duration_seconds: float
    predicted_trip_duration_minutes: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# 5. FastAPI app
app = FastAPI(
    title='NYC Taxi Trip Duration Prediction API',
    description='Predicts taxi trip duration (seconds) using the Random Forest model trained above.',
    version='1.0.0'
)

@app.get('/', tags=['meta'])
def root():
    return {'message': 'Taxi ETA Prediction API', 'docs': '/docs', 'predict': 'POST /predict'}

@app.get('/health', response_model=HealthResponse, tags=['meta'])
def health():
    return HealthResponse(status='ok', model_loaded=rf_model is not None)

@app.post('/predict', response_model=TripPredictionResponse, tags=['prediction'])
def predict(trip: TripRequest):
    try:
        x = build_features(trip.model_dump())
        pred_seconds = max(float(rf_model.predict(x)[0]), 0.0)
        distance_val = float(x['distance_km'].values[0]) if 'distance_km' in x else 0.0
        save_log(distance_val, trip.passenger_count, pred_seconds)
        return TripPredictionResponse(
            predicted_trip_duration_seconds=round(pred_seconds, 2),
            predicted_trip_duration_minutes=round(pred_seconds / 60, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Prediction failed: {e}')

print('FastAPI app defined.')

# 6. Run the API server inside the notebook
nest_asyncio.apply()

def run_server():
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)
print('API running at http://127.0.0.1:8000 (docs: http://127.0.0.1:8000/docs)')

# 7. Test the live API
BASE_URL = 'http://127.0.0.1:8000'
print("Health Check:", requests.get(f'{BASE_URL}/health').json())

trip = {
    'vendor_id': 2,
    'pickup_datetime': '2016-03-14T17:24:55',
    'passenger_count': 1,
    'pickup_longitude': -73.982155,
    'pickup_latitude': 40.767937,
    'dropoff_longitude': -73.964630,
    'dropoff_latitude': 40.765602,
    'store_and_fwd_flag': 'N',
}

response = requests.post(f'{BASE_URL}/predict', json=trip)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())

print("Current Working Directory:", os.getcwd())
print("Files here:", os.listdir())


# Finding the exact location of prediction_logs.csv on the computer
for root, dirs, files in os.walk("C:\\"):
    if "prediction_logs.csv" in files:
        print("File found at path:", os.path.join(root, "prediction_logs.csv"))
        break
