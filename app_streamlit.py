# app_streamlit.py
# --------------------------
# Imports
# --------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import gdown
from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]
import plotly.graph_objects as go
import plotly.express as px
import time

# --------------------------
# Google Drive file links 
# --------------------------
FILES = {
    "lstm": "https://drive.google.com/uc?id=13OBdi-tM95IgInh2tVkF0x1xwfEgQw45",
    "xgb": "https://drive.google.com/uc?id=1TkCgpfeEBgeIXcTk-qMBVbBv9MwDL7_f",
    "preprocessor": "https://drive.google.com/uc?id=10-P-HNOEk-DDTFZmYFkZ9xMgtsy9rKeN"
}

# --------------------------
# Utility: load & cache models/preprocessor
# --------------------------
@st.cache_resource
def load_assets():
    os.makedirs("models", exist_ok=True)
    # download only if missing
    if not os.path.exists("models/energy_model_lstm.keras"):
        gdown.download(FILES["lstm"], "models/energy_model_lstm.keras", quiet=True)
    if not os.path.exists("models/energy_model_xgb.pkl"):
        gdown.download(FILES["xgb"], "models/energy_model_xgb.pkl", quiet=True)
    if not os.path.exists("models/preprocessor.pkl"):
        gdown.download(FILES["preprocessor"], "models/preprocessor.pkl", quiet=True)

    # load safely (wrap in try/except in case files missing)
    model_lstm = None
    model_xgb = None
    preprocessor = None
    try:
        model_lstm = load_model("models/energy_model_lstm.keras", compile=False)
    except Exception:
        model_lstm = None
    try:
        model_xgb = joblib.load("models/energy_model_xgb.pkl")
    except Exception:
        model_xgb = None
    try:
        preprocessor = joblib.load("models/preprocessor.pkl")
    except Exception:
        preprocessor = None

    return model_lstm, model_xgb, preprocessor


model_lstm, model_xgb, preprocessor = load_assets()

# --------------------------
# Page config + styles
# --------------------------
st.set_page_config(page_title="Smart Energy Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center'> Smart Energy Consumption Dashboard</h1>", unsafe_allow_html=True)

# Glassy thick-border CSS for cards
st.markdown(
    """
<style>
.card {
  background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  padding: 16px;
  border-radius: 14px;
  margin: 6px 6px;
  border: 3px solid rgba(0,255,255,0.25);
  backdrop-filter: blur(6px);
  box-shadow: 0 6px 18px rgba(0,0,0,0.45);
  color: #e9f6ff;
}
.card h4 { margin: 0; color: #00e5ff; font-weight:700; }
.card .value { font-size:20px; font-weight:700; margin-top:8px; color:#ffffff; }
.small-note { font-size:12px; color:#b8f1ff; margin-top:6px; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------
# Sidebar: options
# --------------------------
st.sidebar.header("Settings")
model_choice = st.sidebar.radio("Prediction model", options=["XGBoost", "LSTM"])
live_sim = st.sidebar.checkbox("Enable animated live line (safe: up to 500 rows)", value=False)

# Carbon adjust buttons (+ / -)
if "carbon_factor" not in st.session_state:
    st.session_state.carbon_factor = 0.233  # default kgCO₂ per kWh 

st.sidebar.markdown("**Carbon factor (kgCO₂ per kWh)**")
c1, c2 = st.sidebar.columns([1, 1])

# Use plain ASCII "-" and "+"
if c1.button("-"):
    st.session_state.carbon_factor = max(0.0, round(st.session_state.carbon_factor - 0.01, 4))
if c2.button("+"):
    st.session_state.carbon_factor = round(st.session_state.carbon_factor + 0.01, 4)

st.sidebar.write(f"Current factor: **{st.session_state.carbon_factor:.4f} kgCO₂/kWh**")


uploaded_file = st.file_uploader("Upload CSV dataset for prediction", type=["csv"])

# --------------------------
# Expected features used during training (keeps mapping stable)
# --------------------------
EXPECTED_FEATURES = [
    "gen [kW]", "House overall [kW]", "Dishwasher [kW]", "Furnace 1 [kW]",
    "Furnace 2 [kW]", "Home office [kW]", "Fridge [kW]", "Wine cellar [kW]",
    "Garage door [kW]", "Kitchen 12 [kW]", "Kitchen 14 [kW]", "Kitchen 38 [kW]",
    "Barn [kW]", "Well [kW]", "Microwave [kW]", "Living room [kW]", "Solar [kW]",
    "temperature", "humidity", "visibility", "apparentTemperature", "pressure",
    "windSpeed", "windBearing", "precipIntensity", "dewPoint", "precipProbability"
]

# Helper to try to intelligently map similarly named columns
def map_features_from_uploaded(df: pd.DataFrame) -> pd.DataFrame:
    # build lowercase map for easy matching
    lower_map = {col.lower(): col for col in df.columns}
    out = {}
    for feat in EXPECTED_FEATURES:
        # try exact match first
        if feat in df.columns:
            out[feat] = df[feat]
            continue
        # try lowercase match
        lk = feat.lower()
        if lk in lower_map:
            out[feat] = df[lower_map[lk]]
            continue
        # try partial heuristic: remove non-alphanum and compare tokens
        feat_tokens = "".join(ch for ch in feat if ch.isalnum()).lower()
        found = None
        for col in df.columns:
            col_tokens = "".join(ch for ch in col if ch.isalnum()).lower()
            if feat_tokens in col_tokens or col_tokens in feat_tokens:
                found = col
                break
        if found:
            out[feat] = df[found]
        else:
            out[feat] = 0.0
    return pd.DataFrame(out)

# --------------------------
# Main: when file uploaded
# --------------------------
if uploaded_file:
    with st.spinner("Reading dataset..."):
        try:
            df_raw = pd.read_csv(uploaded_file, low_memory=False)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

    st.markdown("### Uploaded data preview")
    st.dataframe(df_raw.head(5))

    # Map features & prepare data for preprocessor/model
    df_mapped = map_features_from_uploaded(df_raw)
    # keep a copy of original mapped features for device charts
    df_features_copy = df_mapped.copy()

    # Clean numeric
    df_mapped.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_mapped.fillna(0, inplace=True)

    # Preprocess & predict
    if preprocessor is None:
        st.warning("Preprocessor not found or failed to load. Predictions will use raw features instead.")
        X = df_mapped.values
    else:
        try:
            X = preprocessor.transform(df_mapped)
        except Exception as e:
            st.error(f"Preprocessor transform failed: {e}")
            st.stop()

    # Predict using chosen model (XGBoost by default)
    if model_choice == "LSTM":
        if model_lstm is None:
            st.error("LSTM model not available. Choose XGBoost or ensure model files exist.")
            st.stop()
        X_lstm = np.expand_dims(X, axis=1)
        preds = model_lstm.predict(X_lstm, verbose=0)
        preds = preds.flatten()
    else:
        if model_xgb is None:
            st.error("XGBoost model not available. Ensure model file exists or choose LSTM.")
            st.stop()
        preds = model_xgb.predict(X)

    df_mapped["Predicted Energy Usage"] = preds

    # --------------------------
    # Key metrics (cards)
    # --------------------------
    total_energy = df_mapped["Predicted Energy Usage"].sum()
    avg_energy = df_mapped["Predicted Energy Usage"].mean()
    max_energy = df_mapped["Predicted Energy Usage"].max()
    # carbon using session factor
    total_carbon = total_energy * st.session_state.carbon_factor
    avg_carbon = avg_energy * st.session_state.carbon_factor
    est_cost = total_energy * 0.12  # example rate

    # cards row (top)
    c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1])
    c1.markdown(f"<div class='card'><h4>Total Energy</h4><div class='value'>{total_energy:,.2f} kW</div><div class='small-note'>sum of predicted usage</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><h4>Average Energy</h4><div class='value'>{avg_energy:,.2f} kW</div><div class='small-note'>mean per row</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><h4>Max Energy</h4><div class='value'>{max_energy:,.2f} kW</div><div class='small-note'>peak predicted</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'><h4>Total Carbon</h4><div class='value'>{total_carbon:,.2f} kgCO₂</div><div class='small-note'>factor: {st.session_state.carbon_factor:.4f} kgCO₂/kWh</div></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='card'><h4>Avg Carbon</h4><div class='value'>{avg_carbon:,.4f} kgCO₂</div><div class='small-note'>avg per row</div></div>", unsafe_allow_html=True)
    c6.markdown(f"<div class='card'><h4>Estimated Cost</h4><div class='value'>${est_cost:,.2f}</div><div class='small-note'>@ $0.12 per kWh</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------
    # Energy Insights cards with embedded charts
    # --------------------------
    st.header("Energy Insights")

    # Peak usage rows + values
    top_n = 5
    peak = df_mapped["Predicted Energy Usage"].nlargest(top_n)
    peak_idx = list(peak.index)
    peak_vals = list(peak.values)

    # Top devices (use original mapped features excluding 'Predicted Energy Usage')
    device_cols = [c for c in df_features_copy.columns if c not in ["Predicted Energy Usage"]]
    # If there are many, pick the ones that match the usage-like columns (we already used mapping)
    device_sums = df_features_copy.sum().sort_values(ascending=False)[:10]

    insight_col1, insight_col2, insight_col3 = st.columns(3)

    # Peak Usage Card (line inside)
    with insight_col1:
        st.markdown("<div class='card'><h4>Peak Usage Times</h4></div>", unsafe_allow_html=True)
        fig_peak = go.Figure()
        fig_peak.add_trace(go.Scatter(x=peak_idx, y=peak_vals, mode="lines+markers",
                                     line=dict(color="crimson", width=4),
                                     marker=dict(size=8, color="orange")))
        fig_peak.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                               xaxis_title="Row index", yaxis_title="Predicted kW")
        st.plotly_chart(fig_peak, use_container_width=True)
        st.write("Rows:", peak_idx)
        st.write("Values:", [f"{v:.2f} kW" for v in peak_vals])

    # Top Devices Card (bar inside)
    with insight_col2:
        st.markdown("<div class='card'><h4>Top Devices Contributing</h4></div>", unsafe_allow_html=True)
        if len(device_sums) > 0:
            fig_devices = px.bar(x=device_sums.index, y=device_sums.values,
                                 color=device_sums.values, color_continuous_scale="Viridis", text=device_sums.values)
            fig_devices.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), yaxis_title="kW")
            st.plotly_chart(fig_devices, use_container_width=True)
        else:
            st.write("No device columns found.")

    # Recommendations Card (mini bar)
    with insight_col3:
        st.markdown("<div class='card'><h4>Energy-Saving Recommendations</h4></div>", unsafe_allow_html=True)
        # simple heuristic: recommend reducing top 3 devices by 10%
        top3 = device_sums.head(3)
        rec_df = pd.DataFrame({"device": top3.index, "current_kW": top3.values})
        rec_df["recommended_kW"] = rec_df["current_kW"] * 0.9
        fig_rec = px.bar(rec_df, x="device", y=["current_kW", "recommended_kW"], barmode="group",
                         labels={"value":"kW","variable":""})
        fig_rec.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_rec, use_container_width=True)
        st.markdown("**Suggestion:** Reduce usage by ~10% on top-consuming devices during peak hours.")

    st.markdown("---")

    # --------------------------
    # Carbon footprint time series (card)
    # --------------------------
    st.subheader("Carbon Footprint Over Time")
    carbon_series = df_mapped["Predicted Energy Usage"] * st.session_state.carbon_factor
    fig_carbon = go.Figure()
    fig_carbon.add_trace(go.Scatter(y=carbon_series, mode="lines", line=dict(color="orange", width=3)))
    fig_carbon.update_layout(template="plotly_dark", xaxis_title="Time", yaxis_title="kgCO₂", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_carbon, use_container_width=True)

    st.markdown("---")

    # --------------------------
    # Animated (or static) line chart for Predicted Energy Usage
    # IMPORTANT: animate only up to max_rows to avoid blocking the page
    # --------------------------
    st.subheader("Predicted Energy Usage Over Time")
    if live_sim:
        line_placeholder = st.empty()
        window_size = 40
        data_buf = []
        max_rows = min(500, len(df_mapped))
        # Show immediate static snapshot before animation to avoid blank page
        fig_initial = go.Figure()
        fig_initial.add_trace(go.Scatter(y=df_mapped["Predicted Energy Usage"].iloc[: min(40, len(df_mapped))],
                                         mode="lines+markers", line=dict(color="cyan", width=3)))
        fig_initial.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10))
        line_placeholder.plotly_chart(fig_initial, use_container_width=True)

        # Animate (non-blocking for other charts because other charts already rendered)
        for i in range(max_rows):
            data_buf.append(df_mapped["Predicted Energy Usage"].iloc[i])
            if len(data_buf) > window_size:
                data_buf = data_buf[-window_size:]
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(y=data_buf, mode="lines+markers",
                                          line=dict(color="cyan", width=3),
                                          marker=dict(size=6, color="lime")))
            fig_line.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10))
            line_placeholder.plotly_chart(fig_line, use_container_width=True)
            time.sleep(0.05)
    else:
        # static full chart
        fig_line_static = go.Figure()
        fig_line_static.add_trace(go.Scatter(y=df_mapped["Predicted Energy Usage"], mode="lines+markers",
                                            line=dict(color="cyan", width=2)))
        fig_line_static.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_line_static, use_container_width=True)

    st.markdown("---")

    # --------------------------
    # Top devices (full) bar chart
    # --------------------------
    st.subheader("Top 10 Energy-Consuming Devices (Full)")
    top_devices_full = df_features_copy.sum().sort_values(ascending=False)[:10]
    fig_bar = px.bar(x=top_devices_full.index, y=top_devices_full.values,
                     color=top_devices_full.values, color_continuous_scale="Viridis", text=top_devices_full.values)
    fig_bar.update_layout(template="plotly_dark", yaxis_title="kW", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)

    # --------------------------
    # Pie chart (distribution)
    # --------------------------
    st.subheader("Energy Distribution Among Top Devices")
    fig_pie = px.pie(values=top_devices_full.values, names=top_devices_full.index, hole=0.4,
                     color=top_devices_full.index, color_discrete_sequence=px.colors.sequential.Viridis)
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

    # --------------------------
    # Download predictions CSV
    # --------------------------
    st.markdown("---")
    csv_out = df_mapped.to_csv(index=False).encode("utf-8")
    st.download_button(" Download Predictions (CSV)", data=csv_out, file_name="predicted_output.csv", mime="text/csv")

else:
    st.info("Upload a CSV to start. The app will map columns heuristically to the model's expected features, then predict and show visuals.")


















