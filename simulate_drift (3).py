import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"

print("--- Starting Drift Simulation (Rush Hour / Festival Surge) ---")

for i in range(3):
    normal_payload = {
        "dropoff_latitude": 40.765602,
        "dropoff_longitude": -73.96463,
        "passenger_count": 2,
        "pickup_datetime": "2016-03-14T10:00:00",
        "pickup_latitude": 40.767937,
        "pickup_longitude": -73.982155,
        "store_and_fwd_flag": "N",
        "vendor_id": 2
    }
    response = requests.post(API_URL, json=normal_payload)
    print(f"Normal Request {i+1} Response:", response.json())

print("\n[!] Simulating Festival / Rush-Hour Surge Drift...")
for i in range(3):
    surge_payload = {
        "dropoff_latitude": 40.850000,
        "dropoff_longitude": -73.900000,
        "passenger_count": 4,
        "pickup_datetime": "2016-03-14T18:30:00",
        "pickup_latitude": 40.712800,
        "pickup_longitude": -74.006000,
        "store_and_fwd_flag": "N",
        "vendor_id": 1
    }
    response = requests.post(API_URL, json=surge_payload)
    print(f"Surge Drift Request {i+1} Response:", response.json())

print("\n--- Drift Simulation Completed Successfully! ---")