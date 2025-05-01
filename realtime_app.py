import streamlit as st
import pandas as pd
import os
import subprocess
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time
from src.data_processing import preprocess

def load_best_model():
    return load_model('outputs/models/best_model.h5')

model = load_best_model()


st.title("🛡️🔵 Intrusion Detection App")

# User selection
mode = st.radio("Choose Input Mode:", ["📂 Upload CSV", "📡 Real-Time Capture"])

if mode == "📂 Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(input_data.head())
        
        input_data_processed = preprocess(input_data)

        if st.button("🔍 Predict Uploaded Data"):

            with st.spinner("Predicting... Please wait ⏳"):
                # Predict
                predictions = model.predict(input_data_processed)
                labels = ["Malicious" if pred > 0.5 else "Benign" for pred in predictions]
                confidence_scores = [float(pred) if pred > 0.5 else 1 - float(pred) for pred in predictions]
                # Create results DataFrame
                results = pd.DataFrame({
                    "Prediction": labels,
                    "Confidence": [f"{conf:.2f}" for conf in confidence_scores]
                })

                st.success("✅ Prediction Completed!")
            
                # Display
                st.subheader("🧾 Prediction Results")
                st.dataframe(results)

                # Metrics if real labels exist
                if 'Label' in input_data.columns:
                    true_labels = input_data['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1).values
                    predicted_labels = [1 if pred > 0.5 else 0 for pred in predictions]

                    st.subheader("📊 Evaluation Metrics")

                    # Accuracy
                    acc = accuracy_score(true_labels, predicted_labels)
                    st.metric("Accuracy", f"{acc*100:.2f}%")

                    # Confusion Matrix
                    cm = confusion_matrix(true_labels, predicted_labels)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)

                    # Classification Report
                    report = classification_report(true_labels, predicted_labels, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.background_gradient(cmap="Blues"))


            pass

elif mode == "📡 Real-Time Capture":
    if st.button("🚀 Start Real-Time Simulation"):
        with st.spinner("Capturing network traffic..."):
            try:
                # 1. Capture packets for 10 seconds into a pcap file
                subprocess.run(
                    "tshark -i en0 -T fields -e frame.time -e ip.src -e ip.dst -e tcp.port -e udp.port -e _ws.col.Info -E separator=, -a duration:10 -w output.pcap",
                    shell=True,
                    check=True
                )
                # 2. Convert pcap to CSV
                subprocess.run(
                    "tshark -r output.pcap -T fields -e frame.time -e ip.src -e ip.dst -e frame.len -E separator=, > network_data.csv",
                    shell=True,
                    check=True
                )

                # 3. Read, preprocess, and predict
                df = pd.read_csv("network_data.csv")
                st.success("✅ Capture Complete!")
                st.dataframe(df.head())

                st.subheader("🧪 Predicting...")
                df_processed = preprocess(df)
                predictions = model.predict(df_processed)
                df["Prediction"] = ["Malicious" if p > 0.5 else "Benign" for p in predictions]
                st.dataframe(df[["frame.time", "ip.src", "ip.dst", "frame.len", "Prediction"]])

            except subprocess.CalledProcessError as e:
                st.error("⚠️ Error running tshark. Make sure it's installed and has permission.")

def predict_realtime(input_path, output_path):
    while True:
        try:
            df = pd.read_csv(input_path)
            processed = preprocess(df)
            predictions = model.predict(processed)
            df['Prediction'] = ["Malicious" if p > 0.5 else "Benign" for p in predictions]
            df.to_csv(output_path, index=False)
            print("Predicted batch. Sleeping...")
        except Exception as e:
            print(f"Waiting for new data... {e}")
        time.sleep(10)

if st.sidebar.button("🚀 Start Real-Time Monitoring"):
    st.sidebar.write("⚙️ Running predictions every 10 seconds...")
    for _ in range(3):  # For demo/testing: run 3 times, replace with while True in production
        try:
            df = pd.read_csv("network_data.csv")
            processed = preprocess(df)
            predictions = model.predict(processed)
            df['Prediction'] = ["Malicious" if p > 0.5 else "Benign" for p in predictions]
            df.to_csv("realtime_predictions.csv", index=False)
            time.sleep(10)
        except Exception as e:
            st.sidebar.warning(f"Waiting for data: {e}")

