import pandas as pd
import time
from src.data_processing import preprocess_single_row
from src.model import build_model
from tensorflow.keras.models import load_model
import os

def simulate_real_time(file_path, model_path, interval=2):
    # Load the full dataset
    df = pd.read_csv(file_path)

    # Load the trained model
    model = load_model(model_path)

    for idx, row in df.iterrows():
        print(f"\nProcessing row {idx + 1}/{len(df)}...")

        try:
            # Preprocess one row (custom function needed!)
            row_df = preprocess_single_row(row)

            # Predict
            prediction = model.predict(row_df)[0][0]
            label = "Malicious" if prediction >= 0.5 else "Benign"
            print(f"Prediction: {prediction:.4f} => {label}")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")

        # Simulate delay (real-time feel)
        time.sleep(interval)
