# --------------------------
# Imports
# --------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import gdown
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
import xgboost as xgb

# --------------------------
# Google Drive file links
# --------------------------
FILES = {
    "lstm": "https://drive.google.com/uc?id=13OBdi-tM95IgInh2tVkF0x1xwfEgQw45",
    "xgb": "https://drive.google.com/uc?id=1TkCgpfeEBgeIXcTk-qMBVb9MwDL7_f",
    "preprocessor": "https://drive.google.com/uc?id=10-P-HNOEk-DDTFZmYFkZ9xMgtsy9rKeN",
    "dataset": "https://drive.google.com/uc?id=1CU1qchDOO9t3cCgZcKJpmmcvcewZRa9o"  
}

# --------------------------
# Download & cache assets
# --------------------------
@st.cache_resource
def load_assets():
    if not os.path.exists("energy_model_lstm.keras"):
        gdown.download(FILES["lstm"], "energy_model_lstm.keras", quiet=False)
    if not os.path.exists("energy_model_xgb.pkl"):
        gdown.download(FILES["xgb"], "energy_model_xgb.pkl", quiet=False)
    if not os.path.exists("preprocessor.pkl"):
        gdown.download(FILES["preprocessor"], "preprocessor.pkl", quiet=False)

    model_lstm = load_model("energy_model_lstm.keras", compile=False)
    model_xgb = joblib.load("energy_model_xgb.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model_lstm, model_xgb, preprocessor

# --------------------------
# Load assets
# --------------------------
model_lstm, model_xgb, preprocessor = load_assets()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Smart Energy Dashboard", layout="wide")
st.title(" Smart Energy Consumption Prediction")

st.sidebar.header("Choose Prediction Model")
model_choice = st.sidebar.radio("Select a model:", ["LSTM", "XGBoost"])

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

expected_features = [
    "gen [kW]", "House overall [kW]", "Dishwasher [kW]", "Furnace 1 [kW]",
    "Furnace 2 [kW]", "Home office [kW]", "Fridge [kW]", "Wine cellar [kW]",
    "Garage door [kW]", "Kitchen 12 [kW]", "Kitchen 14 [kW]", "Kitchen 38 [kW]",
    "Barn [kW]", "Well [kW]", "Microwave [kW]", "Living room [kW]", "Solar [kW]",
    "temperature", "humidity", "visibility", "apparentTemperature", "pressure",
    "windSpeed", "windBearing", "precipIntensity", "dewPoint", "precipProbability"
]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(" Uploaded Data Preview:", df.head())

    # Add missing columns with 0
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    # Keep only expected features
    df = df[expected_features]

    # Clean dataset
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Preprocess data
    try:
        X = preprocessor.transform(df)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # Predict
    if model_choice == "LSTM":
        X_lstm = np.expand_dims(X, axis=1)
        preds = model_lstm.predict(X_lstm)
        st.subheader(" LSTM Predictions")
    else:
        preds = model_xgb.predict(X)
        st.subheader("XGBoost Predictions")

    df["Predicted Energy Usage"] = preds.flatten() if model_choice == "LSTM" else preds
    st.dataframe(df.head(20))

    # Download predictions
    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Predictions CSV", data=csv_download,
                       file_name="predicted_output.csv", mime="text/csv")















