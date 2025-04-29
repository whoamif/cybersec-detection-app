# prettier_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time
from src.data_processing import preprocess

# --------- Page Setup ----------
st.set_page_config(
    page_title="🔵 Intrusion Detection Upload App",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS to make it prettier
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .stButton button {
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------- Load Model ----------
@st.cache_resource
def load_best_model():
    return load_model('outputs/models/best_model.h5')

model = load_best_model()

# --------- App Title ----------
st.title("🛡️🔵 Upload CSV for Intrusion Detection")
st.write("Upload a CSV file and detect if there are malicious activities.")

# --------- File Uploader ----------
uploaded_file = st.file_uploader("📂 Choose a CSV file", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(input_data.head())

    # Preprocess
    input_data_processed = preprocess(input_data)

    # Predict Button
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
