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

# Initialize session state variables
if "simulate" not in st.session_state:
    st.session_state.simulate = False

if "start_index" not in st.session_state:
    st.session_state.start_index = 0

# To store the results
if "results" not in st.session_state:
    st.session_state.results = []

st.title("🛡️ Real-Time Intrusion Detection Simulator")

# Start/Stop Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Simulation"):
        st.session_state.simulate = True
        st.session_state.start_index = 0
        st.session_state.results = []  # Reset results on start

with col2:
    if st.button("⏹️ Stop Simulation"):
        st.session_state.simulate = False
        st.session_state.start_index = 0
        # Save results to a CSV file when stopping the simulation
        results_df = pd.DataFrame(st.session_state.results, columns=["Sample", "Label", "Confidence"])
        results_df.to_csv("simulation_results.csv", index=False)
        st.session_state.results = []  # Clear results after saving
        st.success("✅ Simulation Stopped and Results Saved.")

# Output container
output_area = st.empty()

# Simulation logic
if st.session_state.simulate:
    total = len(X)
    end_index = st.session_state.start_index + 100  # Process the next 100 samples

    for i in range(st.session_state.start_index, min(end_index, total)):
        row = X.iloc[i]
        # Reshape row to match input shape expected by the model
        prediction = model.predict(row.values.reshape(1, -1))[0][0]  # Ensure input is 2D

        label = "Malicious" if prediction > 0.5 else "Benign"
        confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

        # Display the result
        output_area.markdown(
            f"🔍 **Sample {i+1}/{total}** → **{label}** ✅ "
            f"(Confidence: `{confidence:.2f}`)"
        )

        # Store results for later saving
        st.session_state.results.append([i+1, label, confidence])

        time.sleep(0.1)  # Simulate delay

    # After 100 samples, stop the simulation automatically
    if end_index >= total:
        st.session_state.simulate = False
        output_area.success("✅ Simulation Complete.")
        # Save results to a CSV file when simulation is complete
        results_df = pd.DataFrame(st.session_state.results, columns=["Sample", "Label", "Confidence"])
        results_df.to_csv("simulation_results.csv", index=False)
        st.session_state.results = []  # Clear results after saving
