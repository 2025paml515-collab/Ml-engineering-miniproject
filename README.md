# Ml-engineering-miniproject
# NYC Taxi Trip Duration Prediction & MLOps Pipeline

An end-to-end Machine Learning Engineering project implementing model training, FastAPI deployment, logging, drift simulation, and automated retraining triggers.

## Project Structure
- `main.py`: FastAPI server handling prediction requests and logging inputs/outputs.
- `retrain_trigger.py`: Monitors model performance using logs and checks for data/concept drift.
- `simulate_drift.py`: Simulates live rush-hour and festival surge traffic to test the API and logging mechanism.
- `M2_Taxi_Trip_Pipline.ipynb`: Data engineering, feature engineering, and modeling notebook.

## How to Run the Project

### 1. Start the FastAPI Server
Open a terminal/PowerShell and run:
```bash
python main.py

### Run Drift Simulation & Test API
python simulate_drift.py

### Check Retraining Trigger
python retrain_trigger.py
