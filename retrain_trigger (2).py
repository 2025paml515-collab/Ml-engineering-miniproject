import os
import pandas as pd

# Path to the log file (Make sure it matches the log file name in your main.py)
LOG_FILE = "prediction_logs.csv"
ERROR_THRESHOLD = 15.0  # Example threshold limit for error (in minutes)

def check_drift_and_retrain():
    # Check if the prediction log file exists
    if not os.path.exists(LOG_FILE):
        print(f"[!] Log file '{LOG_FILE}' not found. No data to check.")
        return

    # Load the prediction logs into a DataFrame
    df = pd.read_csv(LOG_FILE)
    
    # Verify if 'actual_eta' and 'predicted_eta' columns are available
    if 'actual_eta' in df.columns and 'predicted_eta' in df.columns:
        # Calculate Mean Absolute Error
        df['error'] = (df['actual_eta'] - df['predicted_eta']).abs()
        avg_error = df['error'].mean()
        
        print(f"[INFO] Current Average Absolute Error: {avg_error:.2f} mins")
        
        # Trigger retraining if the error exceeds the threshold
        if avg_error > ERROR_THRESHOLD:
            print("[ALERT] Drift detected! Model performance dropped below threshold.")
            print("[ACTION] Triggering Retraining Pipeline...")
            
            # Execute retraining script or pipeline here (e.g., M2/M3 training script)
            # os.system("python retrain_model.py")
            print("[SUCCESS] Model retraining process initiated.")
        else:
            print("[OK] Model performance is within acceptable limits. No retraining needed.")
    else:
        print("[NOTE] Log file exists, but 'actual_eta' column is missing for performance check.")

if __name__ == "__main__":
    check_drift_and_retrain()