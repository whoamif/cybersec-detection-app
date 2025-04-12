# app.py
import streamlit as st
import pandas as pd
import time
from tensorflow.keras.models import load_model
from src.data_processing import preprocess

# Load model
model = load_model("outputs/models/best_model.h5")

# Load data
df = pd.read_csv("data/CICIDS2017_improved/full_dataset.csv")
df = preprocess(df)

X = df.drop('Label', axis=1)
y = df['Label']

# Title
st.title("🛡️ Real-Time Intrusion Detection Simulator")

# Start simulation
if st.button("Start Simulation"):
    st.write("Starting real-time detection...")
    progress_bar = st.progress(0)
    results = []

    for i, row in X.iterrows():
        prediction = model.predict([row])[0][0]  # assuming binary classification
        label = "Malicious" if prediction > 0.5 else "Benign"
        results.append((i, label))

        st.write(f"Sample {i}: {label}")
        time.sleep(0.1)  # simulate delay
        progress_bar.progress((i + 1) / len(X))

        if i > 100:  # Limit for demo
            break

    st.success("Simulation finished!")
