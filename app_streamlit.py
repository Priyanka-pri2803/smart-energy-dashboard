import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]

# --------------------------
# Load model & preprocessor
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model = load_model(os.path.join(BASE_DIR, "energy_model_lstm.keras"), compile=False)
    preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
    return model, preprocessor

model, preprocessor = load_assets()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Smart Energy Dashboard", layout="wide")
st.title("⚡ Smart Energy Dashboard")
st.write("Upload your dataset or enter values manually to predict **energy usage**.")

# Sidebar for navigation
menu = st.sidebar.radio("Choose an option", ["📂 Upload CSV", "✍️ Manual Input", "📊 About"])

# --------------------------
# 1. CSV Upload Prediction
# --------------------------
if menu == "📂 Upload CSV":
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    # Google Drive direct download link for full dataset
    dataset_url = "https://drive.google.com/uc?export=download&id=1CU1qchDOO9t3cCgZcKJpmmcvcewZRa9o" 
    predicted_dataset_url = "https://drive.google.com/uc?export=download&id=1Cx7Tyh5-ttWJ_Pf_pP6Spr0HRKjZI5_V" 
    predicted_output_url = "https://drive.google.com/uc?export=download&id=1l_dx9o92bBfb_lr30R28mPTIROHbB84Q"


    try:
        if uploaded_file is not None:
            # Use uploaded file
            df = pd.read_csv(uploaded_file)
        else:
            # Download full dataset automatically
            df = pd.read_csv(dataset_url)
            st.info("📥 Using full dataset from Google Drive")

        # --------------------------
        # Keep only expected columns
        # --------------------------
        expected_features = [
            "gen [kW]", "House overall [kW]", "Dishwasher [kW]", "Furnace 1 [kW]",
            "Furnace 2 [kW]", "Home office [kW]", "Fridge [kW]", "Wine cellar [kW]",
            "Garage door [kW]", "Kitchen 12 [kW]", "Kitchen 14 [kW]", "Kitchen 38 [kW]",
            "Barn [kW]", "Well [kW]", "Microwave [kW]", "Living room [kW]", "Solar [kW]",
            "temperature", "humidity", "visibility", "apparentTemperature", "pressure",
            "windSpeed", "windBearing", "precipIntensity", "dewPoint", "precipProbability"
        ]

        # Add missing columns with 0
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        # Keep only expected features
        df = df[expected_features]

        # Clean dataset
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # Preprocess
        X = preprocessor.transform(df)

        # Reshape for LSTM
        X = np.expand_dims(X, axis=1)

        preds = model.predict(X)
        df["Predicted Energy Usage"] = preds.flatten()

        st.success("✅ Prediction completed!")
        st.dataframe(df.head(20))

        # Download
        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions CSV", data=csv_download,
                           file_name="predicted_output.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# --------------------------
# 2. Manual Input
# --------------------------
elif menu == "✍️ Manual Input":
    st.subheader("Enter values manually")

    expected_features = [
        "gen [kW]", "House overall [kW]", "Dishwasher [kW]", "Furnace 1 [kW]",
        "Furnace 2 [kW]", "Home office [kW]", "Fridge [kW]", "Wine cellar [kW]",
        "Garage door [kW]", "Kitchen 12 [kW]", "Kitchen 14 [kW]", "Kitchen 38 [kW]",
        "Barn [kW]", "Well [kW]", "Microwave [kW]", "Living room [kW]", "Solar [kW]",
        "temperature", "humidity", "visibility", "apparentTemperature", "pressure",
        "windSpeed", "windBearing", "precipIntensity", "dewPoint", "precipProbability"
    ]

    user_input = {}
    cols = st.columns(3)
    for i, feature in enumerate(expected_features):
        with cols[i % 3]:
            user_input[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Predict Energy Usage"):
        try:
            df_input = pd.DataFrame([user_input])
            X = preprocessor.transform(df_input)
            X = np.expand_dims(X, axis=1)
            pred = model.predict(X)
            st.success(f"⚡ Predicted Energy Usage: **{pred[0][0]:.2f} kW**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --------------------------
# 3. About
# --------------------------
elif menu == "📊 About":
    st.subheader("About this Project")
    st.write("""
    This Smart Energy Dashboard is built using:
    - **TensorFlow (LSTM)** for time-series prediction
    - **Streamlit** for interactive UI
    - **Preprocessing pipeline** with scaling/encoding
    - **Downloadable predictions** for datasets
    """)











